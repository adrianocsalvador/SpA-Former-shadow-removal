import os
import random
import shutil
import yaml
from attrdict import AttrMap
import time
import cv2
import numpy as np

import torch
from torch import nn
from torch.backends import cudnn
from torch import optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn import functional as F

from data_manager import TrainDataset, ValDataset
from SpA_Former import Generator
from models.dis.dis import Discriminator
import utils
from utils import gpu_manage, save_image, checkpoint
from eval import test
from log_report import LogReport
from log_report import TestReport


class SlidingWindowDataset:
    """Dataset que cria janelas deslizantes em tempo real a partir de imagens originais"""
    
    def __init__(self, config, window_size=(640, 480), stride=(320, 240)):
        self.config = config
        self.window_size = window_size  # (width, height)
        self.stride = stride  # (width, height)
        
        # Carregar lista de imagens
        train_list_file = os.path.join(config.datasets_dir, config.train_list)
        if not os.path.exists(train_list_file) or os.path.getsize(train_list_file) == 0:
            files = os.listdir(os.path.join(config.datasets_dir, 'train_C'))
            n_train = len(files)
            train_list = files[1280:n_train]
            np.savetxt(os.path.join(config.datasets_dir, config.train_list), np.array(train_list), fmt='%s')
        
        self.imlist = np.loadtxt(train_list_file, str)
        self.current_image_idx = 0
        self.current_windows = []
        self._generate_windows_for_current_image()
    
    def _generate_windows_for_current_image(self):
        """Gera todas as janelas para a imagem atual"""
        if self.current_image_idx >= len(self.imlist):
            return
        
        # Carregar imagem original
        img_name = self.imlist[self.current_image_idx]
        img_path = os.path.join(self.config.datasets_dir, 'train_A', img_name)
        target_path = os.path.join(self.config.datasets_dir, 'train_C', img_name)
        
        # Verificar se as imagens existem
        if not os.path.exists(img_path) or not os.path.exists(target_path):
            self.current_image_idx += 1
            self._generate_windows_for_current_image()
            return
        
        # Carregar imagens
        img = cv2.imread(img_path, 1).astype(np.float32)
        target = cv2.imread(target_path, 1).astype(np.float32)
        
        if img is None or target is None:
            self.current_image_idx += 1
            self._generate_windows_for_current_image()
            return
        
        h, w = img.shape[:2]
        
        # Gerar coordenadas das janelas
        windows = []
        for y in range(0, h - self.window_size[1] + 1, self.stride[1]):
            for x in range(0, w - self.window_size[0] + 1, self.stride[0]):
                windows.append((x, y))
        
        # Adicionar janela final se não cobrir toda a imagem
        if h > self.window_size[1]:
            y = h - self.window_size[1]
            for x in range(0, w - self.window_size[0] + 1, self.stride[0]):
                windows.append((x, y))
        
        if w > self.window_size[0]:
            x = w - self.window_size[0]
            for y in range(0, h - self.window_size[1] + 1, self.stride[1]):
                windows.append((x, y))
        
        # Adicionar canto final se necessário
        if h > self.window_size[1] and w > self.window_size[0]:
            windows.append((w - self.window_size[0], h - self.window_size[1]))
        
        # Remover duplicatas
        windows = list(set(windows))
        
        # Armazenar janelas com dados da imagem
        self.current_windows = []
        for x, y in windows:
            # Extrair janelas
            img_window = img[y:y+self.window_size[1], x:x+self.window_size[0]]
            target_window = target[y:y+self.window_size[1], x:x+self.window_size[0]]
            
            # Calcular máscara de sombra
            M = np.clip((target_window - img_window).sum(axis=2), 0, 1).astype(np.float32)
            
            # Normalizar
            img_window = img_window / 255
            target_window = target_window / 255
            
            # Transpor para formato PyTorch (C, H, W)
            img_window = img_window.transpose(2, 0, 1)
            target_window = target_window.transpose(2, 0, 1)
            
            self.current_windows.append((img_window, target_window, M))
    
    def get_next_batch(self, batch_size):
        """Retorna o próximo batch de janelas"""
        batch = []
        
        while len(batch) < batch_size:
            if not self.current_windows:
                # Mover para próxima imagem
                self.current_image_idx += 1
                if self.current_image_idx >= len(self.imlist):
                    # Reiniciar dataset
                    self.current_image_idx = 0
                    random.shuffle(self.imlist)  # Embaralhar ordem das imagens
                
                self._generate_windows_for_current_image()
                
                if not self.current_windows:
                    continue
            
            # Pegar janela da lista atual
            window_data = self.current_windows.pop(0)
            batch.append(window_data)
        
        # Converter para tensores
        imgs = torch.FloatTensor([w[0] for w in batch])
        targets = torch.FloatTensor([w[1] for w in batch])
        masks = torch.FloatTensor([w[2] for w in batch])
        
        return imgs, targets, masks
    
    def __len__(self):
        """Retorna número total de janelas em todas as imagens"""
        total_windows = 0
        for img_name in self.imlist:
            img_path = os.path.join(self.config.datasets_dir, 'train_A', img_name)
            img = cv2.imread(img_path, 1)
            if img is not None:
                h, w = img.shape[:2]
                # Calcular número de janelas
                windows_h = max(1, (h - self.window_size[1]) // self.stride[1] + 1)
                windows_w = max(1, (w - self.window_size[0]) // self.stride[0] + 1)
                total_windows += windows_h * windows_w
        
        return total_windows


def train_with_sliding_window(config):
    """Função principal de treinamento com janela deslizante"""
    gpu_manage(config)

    ### DATASET LOAD ###
    print('===> Loading sliding window dataset')
    
    # Configurar tamanho da janela e stride
    window_size = (config.width, config.height)  # Usar tamanho configurado
    stride = (config.width // 2, config.height // 2)  # Stride de 50%
    
    train_dataset = SlidingWindowDataset(config, window_size, stride)
    validation_dataset = ValDataset(config)
    
    print(f'Sliding window dataset: {len(train_dataset)} total windows')
    print(f'Window size: {window_size}, Stride: {stride}')
    print('validation dataset:', len(validation_dataset))

    validation_data_loader = DataLoader(dataset=validation_dataset, num_workers=config.threads, 
                                       batch_size=config.validation_batchsize, shuffle=False)
    
    ### MODELS LOAD ###
    print('===> Loading models')

    gen = Generator(gpu_ids=config.gpu_ids)

    if config.gen_init is not None:
        param = torch.load(config.gen_init)
        gen.load_state_dict(param)
        print('load {} as pretrained model'.format(config.gen_init))

    dis = Discriminator(in_ch=config.in_ch, out_ch=config.out_ch, gpu_ids=config.gpu_ids)

    if config.dis_init is not None:
        param = torch.load(config.dis_init)
        dis.load_state_dict(param)
        print('load {} as pretrained model'.format(config.dis_init))

    # setup optimizer
    opt_gen = optim.Adam(gen.parameters(), lr=config.lr, betas=(config.beta1, 0.999), weight_decay=0.00001)
    opt_dis = optim.Adam(dis.parameters(), lr=config.lr, betas=(config.beta1, 0.999), weight_decay=0.00001)

    criterionL1 = nn.L1Loss()
    criterionMSE = nn.MSELoss()
    criterionSoftplus = nn.Softplus()

    if config.cuda:
        gen = gen.cuda()
        dis = dis.cuda()
        criterionL1 = criterionL1.cuda()
        criterionMSE = criterionMSE.cuda()
        criterionSoftplus = criterionSoftplus.cuda()

    logreport = LogReport(log_dir=config.out_dir)
    validationreport = TestReport(log_dir=config.out_dir)

    print('===> begin sliding window training')
    start_time = time.time()
    
    # Calcular número de iterações por época
    iterations_per_epoch = len(train_dataset) // config.batchsize
    
    # main training loop
    for epoch in range(1, config.epoch + 1):
        epoch_start_time = time.time()
        
        for iteration in range(1, iterations_per_epoch + 1):
            # Obter batch de janelas deslizantes
            real_a_cpu, real_b_cpu, M_cpu = train_dataset.get_next_batch(config.batchsize)
            
            if config.cuda:
                real_a = real_a_cpu.cuda()
                real_b = real_b_cpu.cuda()
                M = M_cpu.cuda()
            else:
                real_a = real_a_cpu
                real_b = real_b_cpu
                M = M_cpu
            
            real_a = Variable(real_a)
            real_b = Variable(real_b)
            
            att, fake_b = gen.forward(real_a)

            ################
            ### Update D ###
            ################
            
            opt_dis.zero_grad()

            # train with fake
            fake_ab = torch.cat((real_a, fake_b), 1)
            pred_fake = dis.forward(fake_ab.detach())
            batchsize, _, w, h = pred_fake.size()

            loss_d_fake = torch.sum(criterionSoftplus(pred_fake)) / batchsize / w / h

            # train with real
            real_ab = torch.cat((real_a, real_b), 1)
            pred_real = dis.forward(real_ab)
            loss_d_real = torch.sum(criterionSoftplus(-pred_real)) / batchsize / w / h

            # Combined loss
            loss_d = loss_d_fake + loss_d_real

            loss_d.backward()

            if epoch % config.minimax == 0:
                opt_dis.step()

            ################
            ### Update G ###
            ################
            
            opt_gen.zero_grad()

            # First, G(A) should fake the discriminator
            fake_ab = torch.cat((real_a, fake_b), 1)
            pred_fake = dis.forward(fake_ab)
            loss_g_gan = torch.sum(criterionSoftplus(-pred_fake)) / batchsize / w / h

            # Second, G(A) = B
            loss_g_l1 = criterionL1(fake_b, real_b) * config.lamb
            loss_g_att = criterionMSE(att[:,0,:,:], M)
            loss_g = loss_g_gan + loss_g_l1 + loss_g_att

            loss_g.backward()

            opt_gen.step()

            # log
            if iteration % 10 == 0:
                print("===> Epoch[{}]({}/{}): loss_d_fake: {:.4f} loss_d_real: {:.4f} loss_g_gan: {:.4f} loss_g_l1: {:.4f}".format(
                epoch, iteration, iterations_per_epoch, loss_d_fake.item(), loss_d_real.item(), loss_g_gan.item(), loss_g_l1.item()))
                
                log = {}
                log['epoch'] = epoch
                log['iteration'] = iterations_per_epoch * (epoch-1) + iteration
                log['gen/loss'] = loss_g.item()
                log['dis/loss'] = loss_d.item()

                logreport(log)

        print('epoch', epoch, 'finished, use time', time.time() - epoch_start_time)
        
        # Validation
        with torch.no_grad():
            log_validation = test(config, validation_data_loader, gen, criterionMSE, epoch)
            validationreport(log_validation)
        print('validation finished')
        
        # Save checkpoint
        if epoch % config.snapshot_interval == 0:
            checkpoint(config, epoch, gen, dis)

        logreport.save_lossgraph()
        validationreport.save_lossgraph()
    
    print('training time:', time.time() - start_time)


if __name__ == '__main__':
    with open('config.yml', 'r', encoding='UTF-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = AttrMap(config)

    utils.make_manager()
    n_job = utils.job_increment()
    config.out_dir = os.path.join(config.out_dir, '{:06}'.format(n_job))
    os.makedirs(config.out_dir)
    print('Job number: {:04d}'.format(n_job))

    # 保存本次训练时的配置
    shutil.copyfile('config.yml', os.path.join(config.out_dir, 'config.yml'))

    train_with_sliding_window(config)
