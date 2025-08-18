#!/usr/bin/env python3
"""
Script de Treinamento ULTRA para SpA-Former com otimizações extremas de memória
Especialmente projetado para RTX 4060 (8GB VRAM) com janelas 512x512
"""

import os
import sys
import gc
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import yaml
from datetime import datetime
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# Configurar memória ULTRA
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:128'

# Configurações CUDA
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

def clear_memory():
    """Limpeza agressiva de memória"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    gc.collect()

class SimpleConfig:
    """Configuração simplificada"""
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)

class UltraMemoryDataset(Dataset):
    """Dataset otimizado para memória ULTRA"""
    
    def __init__(self, data_dir, window_size=512, max_images=100):
        self.data_dir = data_dir
        self.window_size = window_size
        self.max_images = max_images
        self.windows = []
        
        print(f"🔍 Carregando dataset de {data_dir}")
        self._load_windows()
        print(f"✅ Dataset carregado: {len(self.windows)} janelas")
    
    def _load_windows(self):
        """Carrega janelas de forma otimizada"""
        shadow_dir = os.path.join(self.data_dir, 'train_A')
        shadow_free_dir = os.path.join(self.data_dir, 'train_C')
        
        if not os.path.exists(shadow_dir) or not os.path.exists(shadow_free_dir):
            print(f"❌ Diretórios não encontrados: {shadow_dir} ou {shadow_free_dir}")
            return
        
        # Listar imagens e filtrar por tamanho (remover arquivos muito pequenos)
        shadow_files = []
        shadow_free_files = []
        
        for f in os.listdir(shadow_dir):
            if f.endswith(('.jpg', '.jpeg', '.png')):
                file_path = os.path.join(shadow_dir, f)
                if os.path.getsize(file_path) > 10000:  # Filtrar arquivos > 10KB
                    shadow_files.append(f)
        
        for f in os.listdir(shadow_free_dir):
            if f.endswith(('.jpg', '.jpeg', '.png')):
                file_path = os.path.join(shadow_free_dir, f)
                if os.path.getsize(file_path) > 10000:  # Filtrar arquivos > 10KB
                    shadow_free_files.append(f)
        
        # Limitar número de imagens
        shadow_files = shadow_files[:self.max_images]
        shadow_free_files = shadow_free_files[:self.max_images]
        
        print(f"📁 Encontradas {len(shadow_files)} imagens válidas com sombra e {len(shadow_free_files)} sem sombra")
        
        # Processar cada par de imagens
        for shadow_file in shadow_files:
            # Usar o mesmo nome de arquivo para ambos os diretórios
            shadow_free_file = shadow_file
            if shadow_free_file not in shadow_free_files:
                continue
                
            shadow_path = os.path.join(shadow_dir, shadow_file)
            shadow_free_path = os.path.join(shadow_free_dir, shadow_free_file)
            
            try:
                # Carregar imagens
                shadow_img = cv2.imread(shadow_path)
                shadow_free_img = cv2.imread(shadow_free_path)
                
                # Debug: verificar se as imagens foram carregadas
                if shadow_img is None:
                    print(f"❌ Falha ao carregar imagem: {shadow_path}")
                    continue
                if shadow_free_img is None:
                    print(f"❌ Falha ao carregar imagem: {shadow_free_path}")
                    continue
                
                if shadow_img is None or shadow_free_img is None:
                    print(f"⚠️ Imagem não carregada: {shadow_file}")
                    continue
                
                # Verificar dimensões
                if shadow_img.shape != shadow_free_img.shape:
                    print(f"⚠️ Dimensões diferentes: {shadow_file} - {shadow_img.shape} vs {shadow_free_img.shape}")
                    continue
                
                # Verificar se tem 3 canais
                if len(shadow_img.shape) != 3 or shadow_img.shape[2] != 3:
                    print(f"⚠️ Imagem sem 3 canais: {shadow_file} - {shadow_img.shape}")
                    continue
                
                # Converter BGR para RGB
                shadow_img = cv2.cvtColor(shadow_img, cv2.COLOR_BGR2RGB)
                shadow_free_img = cv2.cvtColor(shadow_free_img, cv2.COLOR_BGR2RGB)
                
                # Verificar se a imagem tem o tamanho esperado
                h, w = shadow_img.shape[:2]
                if h == self.window_size and w == self.window_size:
                    # A imagem já é do tamanho da janela
                    self.windows.append({
                        'shadow': shadow_img.copy(),
                        'shadow_free': shadow_free_img.copy(),
                        'x': 0, 'y': 0
                    })
                else:
                    # Extrair janelas se a imagem for maior
                    for y in range(0, h - self.window_size + 1, self.window_size // 2):
                        for x in range(0, w - self.window_size + 1, self.window_size // 2):
                            shadow_window = shadow_img[y:y+self.window_size, x:x+self.window_size]
                            shadow_free_window = shadow_free_img[y:y+self.window_size, x:x+self.window_size]
                            
                            # Verificar se a janela tem conteúdo válido
                            if shadow_window.shape == (self.window_size, self.window_size, 3):
                                self.windows.append({
                                    'shadow': shadow_window.copy(),
                                    'shadow_free': shadow_free_window.copy(),
                                    'x': x, 'y': y
                                })
                
                # Limpar memória após cada imagem
                del shadow_img, shadow_free_img
                clear_memory()
                
            except Exception as e:
                print(f"❌ Erro ao processar {shadow_file}: {e}")
                continue
    
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        window = self.windows[idx]
        
        # Verificar se os dados são válidos
        if window['shadow'] is None or window['shadow_free'] is None:
            print(f"⚠️ Dados inválidos no índice {idx}")
            # Retornar dados dummy
            dummy_data = np.zeros((self.window_size, self.window_size, 3), dtype=np.uint8)
            shadow = dummy_data.astype(np.float32) / 255.0
            shadow_free = dummy_data.astype(np.float32) / 255.0
        else:
            # Normalizar para [0, 1]
            shadow = window['shadow'].astype(np.float32) / 255.0
            shadow_free = window['shadow_free'].astype(np.float32) / 255.0
        
        # Verificar dimensões
        if shadow.shape != (self.window_size, self.window_size, 3):
            print(f"⚠️ Dimensões incorretas no índice {idx}: {shadow.shape}")
            # Retornar dados dummy
            dummy_data = np.zeros((self.window_size, self.window_size, 3), dtype=np.uint8)
            shadow = dummy_data.astype(np.float32) / 255.0
            shadow_free = dummy_data.astype(np.float32) / 255.0
        
        # Converter para tensor
        shadow = torch.from_numpy(shadow).permute(2, 0, 1)  # HWC -> CHW
        shadow_free = torch.from_numpy(shadow_free).permute(2, 0, 1)
        
        # Verificar se os tensores têm valores válidos
        if torch.isnan(shadow).any() or torch.isinf(shadow).any():
            print(f"⚠️ Valores NaN/Inf no índice {idx}")
            shadow = torch.zeros(3, self.window_size, self.window_size)
            shadow_free = torch.zeros(3, self.window_size, self.window_size)
        
        return shadow, shadow_free

def load_models(config):
    """Carrega modelos com otimizações ULTRA"""
    print("🔄 Carregando modelos...")
    
    # Importar SpA-Former
    sys.path.append('.')
    from SpA_Former import SpA_former
    from models.dis.dis import Discriminator
    
    # Criar modelos
    generator = SpA_former()
    discriminator = Discriminator(in_ch=config.in_ch, out_ch=config.out_ch, gpu_ids=config.gpu_ids)
    
    # Carregar pesos pré-treinados se disponível
    if hasattr(config, 'gen_init') and os.path.exists(config.gen_init):
        print(f"📥 Carregando pesos do gerador de {config.gen_init}")
        try:
            checkpoint = torch.load(config.gen_init, weights_only=True, map_location='cpu')
            
            # Verificar se os pesos têm prefixo 'gen.gen.'
            if 'gen.gen.conv_in.0.weight' in checkpoint:
                # Remover prefixo 'gen.gen.' dos pesos
                new_state_dict = {}
                for key, value in checkpoint.items():
                    if key.startswith('gen.gen.'):
                        new_key = key[7:]  # Remove 'gen.gen.'
                        new_state_dict[new_key] = value
                generator.load_state_dict(new_state_dict)
                print("✅ Pesos carregados com prefixo removido")
            else:
                generator.load_state_dict(checkpoint)
                print("✅ Pesos carregados diretamente")
        except Exception as e:
            print(f"⚠️ Erro ao carregar pesos pré-treinados: {e}")
            print("✅ Iniciando treinamento do zero")
    
    # Mover para GPU
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    
    print("✅ Modelos carregados com sucesso")
    
    print("✅ Modelos carregados com sucesso")
    return generator, discriminator

def create_loss_functions():
    """Cria funções de loss otimizadas"""
    # Loss básica
    criterion_gan = nn.BCELoss()
    criterion_l1 = nn.L1Loss()
    
    # Mover para GPU
    criterion_gan = criterion_gan.cuda()
    criterion_l1 = criterion_l1.cuda()
    
    return criterion_gan, criterion_l1

def train_epoch_ultra(generator, discriminator, dataloader, optimizers, criterions, epoch, config):
    """Treina uma época com otimizações ULTRA"""
    generator.train()
    discriminator.train()
    
    optimizer_g, optimizer_d = optimizers
    criterion_gan, criterion_l1 = criterions
    
    total_loss_g = 0
    total_loss_d = 0
    batch_count = 0
    
    print(f"  🔄 Processando {len(dataloader)} batches...")
    
    for batch_idx, (shadow, shadow_free) in enumerate(dataloader):
        try:
            # Mover para GPU
            shadow = shadow.cuda()
            shadow_free = shadow_free.cuda()
            
            # Labels
            batch_size = shadow.size(0)
            real_label = torch.ones(batch_size, 1).cuda()
            fake_label = torch.zeros(batch_size, 1).cuda()
            
            # ===== Treinar Discriminador =====
            optimizer_d.zero_grad()
            
            # Real
            real_output = discriminator(shadow_free)
            loss_d_real = criterion_gan(real_output, real_label)
            
            # Fake
            with torch.no_grad():
                fake_output = generator(shadow)
            fake_output_d = discriminator(fake_output.detach())
            loss_d_fake = criterion_gan(fake_output_d, fake_label)
            
            # Loss total do discriminador
            loss_d = (loss_d_real + loss_d_fake) * 0.5
            loss_d.backward()
            optimizer_d.step()
            
            # ===== Treinar Gerador =====
            optimizer_g.zero_grad()
            
            # GAN loss
            fake_output_g = discriminator(fake_output)
            loss_g_gan = criterion_gan(fake_output_g, real_label)
            
            # L1 loss
            loss_g_l1 = criterion_l1(fake_output, shadow_free)
            
            # Loss total do gerador
            loss_g = loss_g_gan + config.lambda_l1 * loss_g_l1
            loss_g.backward()
            optimizer_g.step()
            
            # Acumular losses
            total_loss_g += loss_g.item()
            total_loss_d += loss_d.item()
            batch_count += 1
            
            # Limpeza de memória a cada N batches
            if batch_idx % config.cleanup_frequency == 0:
                clear_memory()
            
            # Progresso
            if batch_idx % 10 == 0:
                print(f"    Batch {batch_idx}/{len(dataloader)} - G: {loss_g.item():.4f}, D: {loss_d.item():.4f}")
                
        except Exception as e:
            print(f"❌ Erro no batch {batch_idx}: {e}")
            clear_memory()
            continue
    
    # Calcular médias
    avg_loss_g = total_loss_g / batch_count if batch_count > 0 else 0
    avg_loss_d = total_loss_d / batch_count if batch_count > 0 else 0
    
    print(f"  ✅ Época {epoch} - G: {avg_loss_g:.4f}, D: {avg_loss_d:.4f}")
    
    return avg_loss_g, avg_loss_d

def save_model(generator, discriminator, epoch, output_dir):
    """Salva modelos com timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    
    # Salvar gerador
    gen_path = os.path.join(output_dir, f'gen_model_epoch_{epoch}_{timestamp}.pth')
    torch.save(generator.state_dict(), gen_path)
    
    # Salvar discriminador
    dis_path = os.path.join(output_dir, f'dis_model_epoch_{epoch}_{timestamp}.pth')
    torch.save(discriminator.state_dict(), dis_path)
    
    print(f"💾 Modelos salvos: {gen_path}, {dis_path}")

def train_panoramic_ultra_memory(config):
    """Função principal de treinamento ULTRA"""
    print("🚀 Iniciando treinamento ULTRA otimizado...")
    
    # Configurar seed
    if hasattr(config, 'manualSeed'):
        torch.manual_seed(config.manualSeed)
        random.seed(config.manualSeed)
        np.random.seed(config.manualSeed)
    
    # Criar diretório de saída
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_dir = f"./results_{timestamp}_ultra"
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Resultados salvos em: {output_dir}")
    
    # Carregar dataset
    dataset = UltraMemoryDataset(
        data_dir=config.datasets_dir,
        window_size=config.window_width,
        max_images=50  # Limitar para teste
    )
    
    if len(dataset) == 0:
        print("❌ Dataset vazio!")
        return
    
    # Criar dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=config.batchsize,
        shuffle=True,
        num_workers=0,  # Sem workers para economizar memória
        pin_memory=False  # Desabilitar pin_memory
    )
    
    # Carregar modelos
    generator, discriminator = load_models(config)
    
    # Criar otimizadores
    optimizer_g = optim.Adam(generator.parameters(), lr=config.lr, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=config.lr, betas=(0.5, 0.999))
    
    # Criar funções de loss
    criterions = create_loss_functions()
    
    # Parâmetros de loss
    if not hasattr(config, 'lambda_l1'):
        config.lambda_l1 = 100.0
    
    # Loop de treinamento
    start_time = time.time()
    
    for epoch in range(1, config.epoch + 1):
        print(f"Época {epoch}/{config.epoch} - Processando...")
        
        epoch_start = time.time()
        
        # Treinar época
        loss_g, loss_d = train_epoch_ultra(
            generator, discriminator, dataloader, 
            (optimizer_g, optimizer_d), criterions, epoch, config
        )
        
        epoch_time = time.time() - epoch_start
        print(f"  ✅ Época {epoch} concluída em {epoch_time:.2f}s")
        
        # Salvar modelo a cada época
        save_model(generator, discriminator, epoch, output_dir)
        
        # Limpeza final da época
        clear_memory()
    
    total_time = time.time() - start_time
    print(f"🎉 Treinamento concluído em {total_time:.2f}s")
    print(f"📁 Resultados salvos em: {output_dir}")

if __name__ == '__main__':
    # Carregar configuração
    config_path = 'config_training_v9_pairs_512x512_ultra_memory.yml'
    
    if not os.path.exists(config_path):
        print(f"❌ Arquivo de configuração não encontrado: {config_path}")
        sys.exit(1)
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    config = SimpleConfig(config_dict)
    train_panoramic_ultra_memory(config)


