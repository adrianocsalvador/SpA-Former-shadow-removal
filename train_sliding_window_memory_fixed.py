#!/usr/bin/env python3
"""
Treinamento com janela deslizante - Versão com gerenciamento de memória
"""

import os
import random
import shutil
import yaml
import time
import cv2
import numpy as np
import argparse
import gc

import torch
from torch import nn
from torch.backends import cudnn
from torch import optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn import functional as F

from data_manager import ValDataset
from SpA_Former import Generator
from models.dis.dis import Discriminator
import utils
from utils import gpu_manage, save_image, checkpoint
from eval import test
from log_report import LogReport
from log_report import TestReport


class SimpleConfig:
    """Configuração simples sem attrdict"""
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)


class SlidingWindowDataset:
    """Dataset que cria janelas deslizantes em tempo real"""
    
    def __init__(self, config, window_size=(640, 480), stride=(320, 240)):
        self.config = config
        self.window_size = window_size
        self.stride = stride
        
        # Carregar lista de imagens
        train_a_dir = os.path.join(config.datasets_dir, 'train_A')
        self.imlist = [f for f in os.listdir(train_a_dir) if f.endswith('.png')]
        
        self.current_image_idx = 0
        self.current_windows = []
        self._generate_windows_for_current_image()
    
    def _generate_windows_for_current_image(self):
        """Gera janelas para a imagem atual"""
        if self.current_image_idx >= len(self.imlist):
            return
        
        img_name = self.imlist[self.current_image_idx]
        img_path = os.path.join(self.config.datasets_dir, 'train_A', img_name)
        target_path = os.path.join(self.config.datasets_dir, 'train_C', img_name)
        
        if not os.path.exists(img_path) or not os.path.exists(target_path):
            self.current_image_idx += 1
            self._generate_windows_for_current_image()
            return
        
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
        
        # Adicionar janelas finais
        if h > self.window_size[1]:
            y = h - self.window_size[1]
            for x in range(0, w - self.window_size[0] + 1, self.stride[0]):
                windows.append((x, y))
        
        if w > self.window_size[0]:
            x = w - self.window_size[0]
            for y in range(0, h - self.window_size[1] + 1, self.stride[1]):
                windows.append((x, y))
        
        if h > self.window_size[1] and w > self.window_size[0]:
            windows.append((w - self.window_size[0], h - self.window_size[1]))
        
        windows = list(set(windows))
        
        # Armazenar janelas
        self.current_windows = []
        for x, y in windows:
            img_window = img[y:y+self.window_size[1], x:x+self.window_size[0]]
            target_window = target[y:y+self.window_size[1], x:x+self.window_size[0]]
            
            M = np.clip((target_window - img_window).sum(axis=2), 0, 1).astype(np.float32)
            
            img_window = img_window / 255
            target_window = target_window / 255
            
            img_window = img_window.transpose(2, 0, 1)
            target_window = target_window.transpose(2, 0, 1)
            
            self.current_windows.append((img_window, target_window, M))
    
    def get_next_batch(self, batch_size):
        """Retorna próximo batch"""
        batch = []
        
        while len(batch) < batch_size:
            if not self.current_windows:
                self.current_image_idx += 1
                if self.current_image_idx >= len(self.imlist):
                    self.current_image_idx = 0
                    random.shuffle(self.imlist)
                
                self._generate_windows_for_current_image()
                
                if not self.current_windows:
                    continue
            
            window_data = self.current_windows.pop(0)
            batch.append(window_data)
        
        # Converter para numpy arrays primeiro para evitar warning
        imgs_array = np.array([w[0] for w in batch])
        targets_array = np.array([w[1] for w in batch])
        masks_array = np.array([w[2] for w in batch])
        
        imgs = torch.FloatTensor(imgs_array)
        targets = torch.FloatTensor(targets_array)
        masks = torch.FloatTensor(masks_array)
        
        return imgs, targets, masks
    
    def __len__(self):
        """Número total de janelas"""
        total = 0
        for img_name in self.imlist:
            img_path = os.path.join(self.config.datasets_dir, 'train_A', img_name)
            img = cv2.imread(img_path, 1)
            if img is not None:
                h, w = img.shape[:2]
                windows_h = max(1, (h - self.window_size[1]) // self.stride[1] + 1)
                windows_w = max(1, (w - self.window_size[0]) // self.stride[0] + 1)
                total += windows_h * windows_w
        return total


def clear_gpu_memory():
    """Força a liberação de memória GPU"""
    print("🧹 Limpando memória GPU...")
    
    # Limpar cache CUDA
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Coleta de lixo
    gc.collect()
    
    # Verificar memória disponível
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"Memória GPU após limpeza: {allocated:.2f}GB alocada, {reserved:.2f}GB reservada")


def train_with_sliding_window(config):
    """Função principal de treinamento"""
    
    # Limpar memória antes de começar
    clear_gpu_memory()
    
    gpu_manage(config)

    print('===> Loading sliding window dataset')
    
    # Configurar tamanho da janela e stride
    window_size = (getattr(config, 'width', 640), getattr(config, 'height', 480))
    stride = (window_size[0] // 2, window_size[1] // 2)
    
    train_dataset = SlidingWindowDataset(config, window_size, stride)
    validation_dataset = ValDataset(config)
    
    print(f'Sliding window dataset: {len(train_dataset)} total windows')
    print(f'Window size: {window_size}, Stride: {stride}')
    print('validation dataset:', len(validation_dataset))

    validation_data_loader = DataLoader(dataset=validation_dataset, num_workers=config.threads, 
                                       batch_size=config.validation_batchsize, shuffle=False)
    
    print('===> Loading models')

    gen = Generator(gpu_ids=config.gpu_ids)

    if hasattr(config, 'gen_init') and config.gen_init:
        param = torch.load(config.gen_init)
        gen.load_state_dict(param)
        print('load {} as pretrained model'.format(config.gen_init))

    dis = Discriminator(in_ch=config.in_ch, out_ch=config.out_ch, gpu_ids=config.gpu_ids)

    if hasattr(config, 'dis_init') and config.dis_init:
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
            # Limpar memória a cada iteração
            if iteration % 5 == 0:
                clear_gpu_memory()
            
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


def main():
    parser = argparse.ArgumentParser(description='SpA-Former Sliding Window Training')
    parser.add_argument('--config', type=str, default='config_sliding_window.yml',
                       help='Caminho para arquivo de configuração')
    parser.add_argument('--window_size', type=int, nargs=2, default=[640, 480],
                       help='Tamanho da janela (width height)')
    parser.add_argument('--stride', type=int, nargs=2, default=[320, 240],
                       help='Stride da janela (width height)')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Tamanho do batch')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Número de épocas')
    parser.add_argument('--lr', type=float, default=0.0004,
                       help='Learning rate')
    
    args = parser.parse_args()
    
    # Carregar configuração
    if not os.path.exists(args.config):
        print(f"Erro: Arquivo de configuração {args.config} não encontrado!")
        return
    
    with open(args.config, 'r', encoding='UTF-8') as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)
    
    # Atualizar configuração com argumentos
    config_dict['window_width'] = args.window_size[0]
    config_dict['window_height'] = args.window_size[1]
    config_dict['stride_width'] = args.stride[0]
    config_dict['stride_height'] = args.stride[1]
    config_dict['batchsize'] = args.batch_size
    config_dict['epoch'] = args.epochs
    config_dict['lr'] = args.lr
    
    config = SimpleConfig(config_dict)
    
    # Verificar dataset
    if not os.path.exists(config.datasets_dir):
        print(f"Erro: Dataset não encontrado em {config.datasets_dir}")
        return
    
    print("=== Configuração do Treinamento ===")
    print(f"Dataset: {config.datasets_dir}")
    print(f"Tamanho da janela: {config.window_width}x{config.window_height}")
    print(f"Stride: {config.stride_width}x{config.stride_height}")
    print(f"Batch size: {config.batchsize}")
    print(f"Épocas: {config.epoch}")
    print(f"Learning rate: {config.lr}")
    print(f"Output directory: {config.out_dir}")
    
    # Criar diretório de saída
    utils.make_manager()
    n_job = utils.job_increment()
    config.out_dir = os.path.join(config.out_dir, '{:06}'.format(n_job))
    os.makedirs(config.out_dir)
    print(f'Job number: {n_job:04d}')
    
    # Salvar configuração
    shutil.copyfile(args.config, os.path.join(config.out_dir, 'config.yml'))
    
    # Executar treinamento
    train_with_sliding_window(config)


if __name__ == '__main__':
    main()

