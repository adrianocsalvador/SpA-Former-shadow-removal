#!/usr/bin/env python3
"""
Treinamento otimizado para imagens panorâmicas com economia máxima de memória
"""

import os
import random
import yaml
import time
import cv2
import numpy as np
import argparse
import math
import gc
import torch

from torch import nn
from torch.backends import cudnn
from torch import optim
from torch.utils.data import DataLoader

from data_manager import ValDataset
from SpA_Former import Generator
from models.dis.dis import Discriminator
import utils
from utils import gpu_manage, save_image, checkpoint
from eval import test
from log_report import LogReport


class MemoryOptimizedDataset:
    """Dataset otimizado para economia máxima de memória"""
    
    def __init__(self, config, window_size=(256, 256), stride=(128, 128)):
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
        """Gera janelas para a imagem atual com otimização de memória"""
        if self.current_image_idx >= len(self.imlist):
            return
        
        img_name = self.imlist[self.current_image_idx]
        img_path = os.path.join(self.config.datasets_dir, 'train_A', img_name)
        target_path = os.path.join(self.config.datasets_dir, 'train_C', img_name)
        
        if not os.path.exists(img_path) or not os.path.exists(target_path):
            self.current_image_idx += 1
            self._generate_windows_for_current_image()
            return
        
        # Carregar imagem com redimensionamento se necessário
        img = cv2.imread(img_path, 1).astype(np.float32)
        target = cv2.imread(target_path, 1).astype(np.float32)
        
        if img is None or target is None:
            self.current_image_idx += 1
            self._generate_windows_for_current_image()
            return
        
        # Redimensionar se a imagem for muito grande
        max_size = getattr(self.config, 'max_image_size', 1024)
        h, w = img.shape[:2]
        if h > max_size or w > max_size:
            scale = min(max_size / h, max_size / w)
            new_h, new_w = int(h * scale), int(w * scale)
            img = cv2.resize(img, (new_w, new_h))
            target = cv2.resize(target, (new_w, new_h))
            print(f"📏 Redimensionado {img_name}: {w}x{h} -> {new_w}x{new_h}")
        
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
        
        # Armazenar janelas com processamento otimizado
        self.current_windows = []
        for x, y in windows:
            img_window = img[y:y+self.window_size[1], x:x+self.window_size[0]]
            target_window = target[y:y+self.window_size[1], x:x+self.window_size[0]]
            
            # Calcular métricas para balanceamento
            shadow_intensity = self._calculate_shadow_intensity(img_window, target_window)
            contrast = self._calculate_contrast(img_window)
            
            # Normalizar
            img_window = img_window / 255
            target_window = target_window / 255
            
            # Transpor para formato PyTorch
            img_window = img_window.transpose(2, 0, 1)
            target_window = target_window.transpose(2, 0, 1)
            
            # Calcular máscara de diferença
            M = np.clip((target_window - img_window).sum(axis=0), 0, 1).astype(np.float32)
            
            self.current_windows.append((img_window, target_window, M, shadow_intensity, contrast))
    
    def _calculate_shadow_intensity(self, img_window, target_window):
        """Calcula intensidade da sombra na janela"""
        diff = np.abs(target_window - img_window)
        return np.mean(diff)
    
    def _calculate_contrast(self, img_window):
        """Calcula contraste da janela"""
        gray = cv2.cvtColor(img_window.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        return np.std(gray)
    
    def get_next_batch(self, batch_size):
        """Retorna próximo batch com otimização de memória"""
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
            
            # Selecionar janela com balanceamento
            if len(batch) < batch_size // 2:
                # Primeira metade: janelas com sombras
                shadow_windows = [w for w in self.current_windows if w[3] > 0.1]
                if shadow_windows:
                    window_data = random.choice(shadow_windows)
                    self.current_windows.remove(window_data)
                else:
                    window_data = self.current_windows.pop(0)
            else:
                # Segunda metade: janelas sem sombras ou com baixo contraste
                low_contrast_windows = [w for w in self.current_windows if w[4] < 30 or w[3] < 0.05]
                if low_contrast_windows:
                    window_data = random.choice(low_contrast_windows)
                    self.current_windows.remove(window_data)
                else:
                    window_data = self.current_windows.pop(0)
            
            batch.append(window_data)
        
        # Converter para tensores
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
                # Redimensionar se necessário
                max_size = getattr(self.config, 'max_image_size', 1024)
                if h > max_size or w > max_size:
                    scale = min(max_size / h, max_size / w)
                    h, w = int(h * scale), int(w * scale)
                windows_h = max(1, (h - self.window_size[1]) // self.stride[1] + 1)
                windows_w = max(1, (w - self.window_size[0]) // self.stride[0] + 1)
                total += windows_h * windows_w
        return total


class AdaptiveLoss(nn.Module):
    """Loss adaptativa que penaliza branqueamento desnecessário"""
    
    def __init__(self, lambda_l1=100, lambda_whitening=10):
        super(AdaptiveLoss, self).__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_whitening = lambda_whitening
        self.l1_loss = nn.L1Loss()
    
    def forward(self, pred, target, input_img):
        # Loss L1 básica
        l1_loss = self.l1_loss(pred, target)
        
        # Loss de branqueamento: penalizar quando pred > target sem necessidade
        whitening_penalty = torch.mean(torch.clamp(pred - target, min=0))
        
        # Loss adaptativa baseada no contraste
        contrast = torch.std(input_img, dim=[2, 3])
        contrast_factor = torch.exp(-contrast * 5)  # Mais peso para baixo contraste
        
        total_loss = l1_loss + self.lambda_whitening * whitening_penalty * contrast_factor.mean()
        
        return total_loss, l1_loss, whitening_penalty


def clear_memory():
    """Limpa memória GPU e CPU"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def train_panoramic_memory_optimized(config):
    """Função principal de treinamento otimizada para memória"""
    gpu_manage(config)

    print('===> Loading panoramic dataset')
    
    # Configurar tamanho da janela e stride
    window_size = (getattr(config, 'window_width', 256), getattr(config, 'window_height', 256))
    stride = (getattr(config, 'stride_width', 128), getattr(config, 'stride_height', 128))
    
    train_dataset = MemoryOptimizedDataset(config, window_size, stride)
    validation_dataset = ValDataset(config)
    
    print(f'Panoramic dataset: {len(train_dataset)} total windows')
    print(f'Window size: {window_size}, Stride: {stride}')
    print('validation dataset:', len(validation_dataset))

    validation_data_loader = DataLoader(dataset=validation_dataset, num_workers=config.threads, 
                                       batch_size=config.validation_batchsize, shuffle=False)
    
    print('===> Loading models')

    gen = Generator(gpu_ids=config.gpu_ids)
    
    # Carregar modelo pré-treinado se especificado
    if hasattr(config, 'gen_init') and config.gen_init:
        param = torch.load(config.gen_init, weights_only=True)
        gen.load_state_dict(param)
        print('load {} as pretrained model'.format(config.gen_init))

    dis = Discriminator(in_ch=config.in_ch, out_ch=config.out_ch, gpu_ids=config.gpu_ids)

    if hasattr(config, 'dis_init') and config.dis_init:
        param = torch.load(config.dis_init, weights_only=True)
        dis.load_state_dict(param)
        print('load {} as pretrained model'.format(config.dis_init))

    # Setup optimizers
    opt_gen = optim.Adam(gen.parameters(), lr=config.lr, betas=(config.beta1, 0.999), weight_decay=0.00001)
    opt_dis = optim.Adam(dis.parameters(), lr=config.lr, betas=(config.beta1, 0.999), weight_decay=0.00001)
    
    # Loss functions
    adaptive_loss = AdaptiveLoss(lambda_l1=config.lamb, lambda_whitening=10)
    criterionMSE = nn.MSELoss()
    criterionSoftplus = nn.Softplus()

    # Mover modelos para GPU
    if config.cuda:
        gen = gen.cuda()
        dis = dis.cuda()
        adaptive_loss = adaptive_loss.cuda()
        criterionMSE = criterionMSE.cuda()
        criterionSoftplus = criterionSoftplus.cuda()

    # Tensors para GPU
    real_a = torch.FloatTensor()
    real_b = torch.FloatTensor()
    M = torch.FloatTensor()

    if config.cuda:
        real_a = real_a.cuda()
        real_b = real_b.cuda()
        M = M.cuda()

    start_time = time.time()
    
    # Gradient accumulation steps
    accumulation_steps = getattr(config, 'gradient_accumulation_steps', 1)
    
    # Main training loop
    for epoch in range(1, config.epoch + 1):
        epoch_start_time = time.time()
        
        # Reset dataset para novo epoch
        train_dataset.current_image_idx = 0
        train_dataset.current_windows = []
        train_dataset._generate_windows_for_current_image()
        
        total_batches = len(train_dataset) // config.batchsize
        opt_gen.zero_grad()
        opt_dis.zero_grad()
        
        for iteration in range(total_batches):
            # Get batch
            real_a_cpu, real_b_cpu, M_cpu = train_dataset.get_next_batch(config.batchsize)
            real_a.resize_(real_a_cpu.size()).copy_(real_a_cpu)
            real_b.resize_(real_b_cpu.size()).copy_(real_b_cpu)
            M.resize_(M_cpu.size()).copy_(M_cpu)
            
            # Forward pass
            att, fake_b = gen.forward(real_a)

            ################
            ### Update D ###
            ################
            
            opt_dis.zero_grad()

            # Train with fake
            fake_ab = torch.cat((real_a, fake_b), 1)
            pred_fake = dis.forward(fake_ab.detach())
            batchsize, _, w, h = pred_fake.size()

            loss_d_fake = torch.sum(criterionSoftplus(pred_fake)) / batchsize / w / h

            # Train with real
            real_ab = torch.cat((real_a, real_b), 1)
            pred_real = dis.forward(real_ab)
            loss_d_real = torch.sum(criterionSoftplus(-pred_real)) / batchsize / w / h

            # Combined loss
            loss_d = loss_d_fake + loss_d_real

            loss_d.backward()
            opt_dis.step()

            ################
            ### Update G ###
            ################
            
            # First, G(A) should fake the discriminator
            fake_ab = torch.cat((real_a, fake_b), 1)
            pred_fake = dis.forward(fake_ab)
            loss_g_gan = torch.sum(criterionSoftplus(-pred_fake)) / batchsize / w / h

            # Second, adaptive loss for better preservation
            loss_g_adaptive, l1_loss, whitening_penalty = adaptive_loss(fake_b, real_b, real_a)
            loss_g_att = criterionMSE(att[:,0,:,:], M)
            
            loss_g = loss_g_gan + loss_g_adaptive + loss_g_att

            # Gradient accumulation
            (loss_g / accumulation_steps).backward()
            if (iteration + 1) % accumulation_steps == 0:
                opt_gen.step()
                opt_gen.zero_grad()

            # Log progress
            if iteration % 10 == 0:
                print(f'Epoch [{epoch}/{config.epoch}] Iteration [{iteration}/{total_batches}] '
                      f'Loss_D: {loss_d.item():.4f} Loss_G: {loss_g.item():.4f} '
                      f'L1: {l1_loss.item():.4f} Whitening: {whitening_penalty.item():.4f}')
            
            # Limpeza de memória a cada 50 iterações
            if iteration % 50 == 0 and getattr(config, 'enable_memory_cleanup', False):
                clear_memory()

        # Validation
        if epoch % config.snapshot_interval == 0:
            test(config, validation_data_loader, gen, criterionMSE, epoch)
            checkpoint(config, epoch, gen, dis, opt_gen, opt_dis)

        epoch_time = time.time() - epoch_start_time
        print(f'Epoch {epoch} completed in {epoch_time:.2f}s')
        
        # Limpeza de memória após cada época
        if getattr(config, 'enable_memory_cleanup', False):
            clear_memory()

    total_time = time.time() - start_time
    print(f'Training completed in {total_time:.2f}s')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config_training_v9_pairs_memory.yml', help='path to config file')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Converter para objeto simples
    class SimpleConfig:
        def __init__(self, config_dict):
            for key, value in config_dict.items():
                setattr(self, key, value)
    
    config = SimpleConfig(config)
    train_panoramic_memory_optimized(config)
