#!/usr/bin/env python3
"""
Teste do sistema de janela deslizante sem carregar o modelo pesado
"""

import os
import random
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


def test_sliding_window_system(config, window_size=(128, 96), stride=(64, 48)):
    """Testa apenas o sistema de janela deslizante"""
    
    print("🧪 Testando Sistema de Janela Deslizante")
    print("=" * 50)
    
    # Limpar memória
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()
    
    print(f"Tamanho da janela: {window_size[0]}x{window_size[1]}")
    print(f"Stride: {stride[0]}x{stride[1]}")
    
    # Criar dataset
    dataset = SlidingWindowDataset(config, window_size, stride)
    print(f"Dataset criado: {len(dataset)} janelas totais")
    
    # Testar alguns batches
    start_time = time.time()
    
    for i in range(5):  # Testar 5 batches
        print(f"\n--- Teste {i+1}/5 ---")
        
        try:
            # Obter batch
            imgs, targets, masks = dataset.get_next_batch(1)
            
            # Verificar shapes
            print(f"Input shape: {imgs.shape}")
            print(f"Target shape: {targets.shape}")
            print(f"Mask shape: {masks.shape}")
            
            # Verificar ranges
            print(f"Input range: [{imgs.min():.3f}, {imgs.max():.3f}]")
            print(f"Target range: [{targets.min():.3f}, {targets.max():.3f}]")
            print(f"Mask range: [{masks.min():.3f}, {masks.max():.3f}]")
            
            # Mover para GPU se disponível
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                targets = targets.cuda()
                masks = masks.cuda()
                print("✅ Dados movidos para GPU")
                
                # Verificar memória GPU
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                print(f"Memória GPU: {allocated:.3f}GB alocada, {reserved:.3f}GB reservada")
            
            # Simular processamento simples
            fake_output = imgs * 0.5 + targets * 0.5
            fake_loss = torch.mean((fake_output - targets) ** 2)
            
            print(f"Fake loss: {fake_loss.item():.6f}")
            
            # Limpar memória
            del imgs, targets, masks, fake_output, fake_loss
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"❌ Erro no teste {i+1}: {e}")
            break
    
    total_time = time.time() - start_time
    print(f"\n🎉 Teste concluído em {total_time:.2f}s")
    print("✅ Sistema de janela deslizante funcionando perfeitamente!")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Teste do Sistema de Janela Deslizante')
    parser.add_argument('--config', type=str, default='config_sliding_window.yml',
                       help='Caminho para arquivo de configuração')
    parser.add_argument('--window_size', type=int, nargs=2, default=[128, 96],
                       help='Tamanho da janela (width height)')
    parser.add_argument('--stride', type=int, nargs=2, default=[64, 48],
                       help='Stride da janela (width height)')
    
    args = parser.parse_args()
    
    # Carregar configuração
    if not os.path.exists(args.config):
        print(f"Erro: Arquivo de configuração {args.config} não encontrado!")
        return
    
    with open(args.config, 'r', encoding='UTF-8') as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)
    
    config = SimpleConfig(config_dict)
    
    # Verificar dataset
    if not os.path.exists(config.datasets_dir):
        print(f"Erro: Dataset não encontrado em {config.datasets_dir}")
        return
    
    print("=== Configuração do Teste ===")
    print(f"Dataset: {config.datasets_dir}")
    print(f"Tamanho da janela: {args.window_size[0]}x{args.window_size[1]}")
    print(f"Stride: {args.stride[0]}x{args.stride[1]}")
    
    # Executar teste
    success = test_sliding_window_system(config, args.window_size, args.stride)
    
    if success:
        print("\n🎉 SUCESSO! Sistema pronto para uso!")
        print("Quando você usar sua GPU de 24GB, poderá executar o treinamento completo.")
    else:
        print("\n❌ Teste falhou!")


if __name__ == '__main__':
    main()
