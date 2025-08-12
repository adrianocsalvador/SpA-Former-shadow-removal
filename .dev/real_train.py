#!/usr/bin/env python3

import os
import sys
import yaml
import argparse
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import time
import cv2
import numpy as np
import shutil
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Configurar backend para funcionar sem display
import matplotlib.pyplot as plt
import re

# Adicionar o diretório raiz ao path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from SpA_Former import Generator
from models.dis.dis import Discriminator
import utils

class SimpleConfig:
    """Classe simples para substituir AttrMap"""
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)

class RealDataset(torch.utils.data.Dataset):
    """Dataset real para carregar imagens com e sem sombra"""
    
    def __init__(self, train_A_dir, train_C_dir, width=320, height=240, max_images=None):
        self.train_A_dir = train_A_dir  # Imagens com sombra
        self.train_C_dir = train_C_dir  # Imagens sem sombra
        self.width = width
        self.height = height
        
        # Listar arquivos
        self.files = []
        for filename in os.listdir(train_A_dir):
            if filename.endswith('.png'):
                if os.path.exists(os.path.join(train_C_dir, filename)):
                    self.files.append(filename)
        
        # Limitar número de imagens se especificado
        if max_images and len(self.files) > max_images:
            import random
            random.shuffle(self.files)
            original_count = len(self.files)
            self.files = self.files[:max_images]
            print(f"📊 Dataset limitado: {len(self.files)} pares de imagens (de {original_count} disponíveis)")
        else:
            print(f"📊 Dataset carregado: {len(self.files)} pares de imagens")
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        filename = self.files[idx]
        
        # Carregar imagem com sombra (entrada)
        img_A_path = os.path.join(self.train_A_dir, filename)
        img_A = cv2.imread(img_A_path)
        if img_A is None:
            raise ValueError(f"Imagem não pode ser carregada: {img_A_path}")
        img_A = cv2.resize(img_A, (self.width, self.height))
        img_A = img_A.astype(np.float32) / 255.0  # Normalizar para [0,1]
        img_A = torch.from_numpy(img_A.transpose(2, 0, 1))  # HWC -> CHW
        
        # Carregar imagem sem sombra (target)
        img_C_path = os.path.join(self.train_C_dir, filename)
        img_C = cv2.imread(img_C_path)
        if img_C is None:
            raise ValueError(f"Imagem não pode ser carregada: {img_C_path}")
        img_C = cv2.resize(img_C, (self.width, self.height))
        img_C = img_C.astype(np.float32) / 255.0  # Normalizar para [0,1]
        img_C = torch.from_numpy(img_C.transpose(2, 0, 1))  # HWC -> CHW
        
        return img_A, img_C

def backup_previous_results(out_dir):
    """Faz backup dos resultados anteriores"""
    if os.path.exists(out_dir):
        # Criar timestamp para o backup
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = f"{out_dir}_backup_{timestamp}"
        
        print(f"📦 Fazendo backup dos dados anteriores...")
        print(f"   De: {out_dir}")
        print(f"   Para: {backup_dir}")
        
        try:
            shutil.move(out_dir, backup_dir)
            print(f"✅ Backup concluído: {backup_dir}")
        except Exception as e:
            print(f"❌ Erro no backup: {e}")
            # Se falhar, tentar renomear
            try:
                os.rename(out_dir, backup_dir)
                print(f"✅ Backup concluído (renomeação): {backup_dir}")
            except Exception as e2:
                print(f"❌ Falha no backup: {e2}")
                return False
    return True

def simple_checkpoint(model, epoch, out_dir, name):
    """Função simplificada para salvar checkpoint"""
    model_dir = os.path.join(out_dir, 'models')
    os.makedirs(model_dir, exist_ok=True)
    model_out_path = os.path.join(model_dir, f'{name}_model_epoch_{epoch}.pth')
    torch.save(model.state_dict(), model_out_path)
    print(f"💾 {name} salvo: {model_out_path}")

def parse_training_logs(log_file_path):
    """Analisa os logs de treinamento e extrai as métricas"""
    epochs = []
    loss_d_values = []
    loss_g_values = []
    loss_g_gan_values = []
    loss_g_l1_values = []
    
    if not os.path.exists(log_file_path):
        return None
    
    with open(log_file_path, 'r') as f:
        content = f.read()
    
    # Padrão para encontrar as métricas com componentes separados
    pattern = r'Época (\d+) - Loss_D: ([\d.]+), Loss_G: ([\d.]+) \(GAN: ([\d.]+), L1: ([\d.]+)\)'
    matches = re.findall(pattern, content)
    
    if not matches:
        # Fallback para formato antigo
        pattern_old = r'Época (\d+) - Loss_D: ([\d.]+), Loss_G: ([\d.]+)'
        matches = re.findall(pattern_old, content)
        if not matches:
            return None
        
        for epoch, loss_d, loss_g in matches:
            epochs.append(int(epoch))
            loss_d_values.append(float(loss_d))
            loss_g_values.append(float(loss_g))
            loss_g_gan_values.append(0.0)  # Valor padrão
            loss_g_l1_values.append(0.0)   # Valor padrão
    else:
        for epoch, loss_d, loss_g, loss_g_gan, loss_g_l1 in matches:
            epochs.append(int(epoch))
            loss_d_values.append(float(loss_d))
            loss_g_values.append(float(loss_g))
            loss_g_gan_values.append(float(loss_g_gan))
            loss_g_l1_values.append(float(loss_g_l1))
    
    return {
        'epochs': epochs,
        'loss_d': loss_d_values,
        'loss_g': loss_g_values,
        'loss_g_gan': loss_g_gan_values,
        'loss_g_l1': loss_g_l1_values
    }

def create_training_plots(data, output_dir):
    """Cria gráficos de evolução do treinamento"""
    if data is None or len(data['epochs']) == 0:
        return
    
    epochs = data['epochs']
    loss_d = data['loss_d']
    loss_g = data['loss_g']
    loss_g_gan = data['loss_g_gan']
    loss_g_l1 = data['loss_g_l1']
    
    # Configurar matplotlib
    plt.style.use('default')
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Evolução do Treinamento SpA-Former', fontsize=16, fontweight='bold')
    
    # Gráfico 1: Evolução da Loss_D
    ax1.plot(epochs, loss_d, 'b-', linewidth=2, label='Loss_D')
    ax1.set_title('Evolução da Loss_D (Discriminator)', fontweight='bold')
    ax1.set_xlabel('Época')
    ax1.set_ylabel('Loss_D')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Gráfico 2: Componentes do Loss_G
    ax2.plot(epochs, loss_g, 'r-', linewidth=2, label='Loss_G (Total)')
    ax2.plot(epochs, loss_g_gan, 'g-', linewidth=2, label='Loss_G_GAN')
    ax2.plot(epochs, loss_g_l1, 'orange', linewidth=2, label='Loss_G_L1')
    ax2.set_title('Componentes do Loss_G (Generator)', fontweight='bold')
    ax2.set_xlabel('Época')
    ax2.set_ylabel('Loss_G')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Gráfico 3: Comparação Loss_D vs Loss_G
    ax3.plot(epochs, loss_d, 'b-', linewidth=2, label='Loss_D')
    ax3.plot(epochs, loss_g, 'r-', linewidth=2, label='Loss_G')
    ax3.set_title('Comparação Loss_D vs Loss_G', fontweight='bold')
    ax3.set_xlabel('Época')
    ax3.set_ylabel('Loss')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Gráfico 4: Análise de Convergência (média móvel)
    if len(epochs) > 5:
        window = min(5, len(epochs) // 4)
        loss_d_smooth = np.convolve(loss_d, np.ones(window)/window, mode='valid')
        loss_g_smooth = np.convolve(loss_g, np.ones(window)/window, mode='valid')
        epochs_smooth = epochs[window-1:]
        
        ax4.plot(epochs_smooth, loss_d_smooth, 'b-', linewidth=2, label='Loss_D (média móvel)')
        ax4.plot(epochs_smooth, loss_g_smooth, 'r-', linewidth=2, label='Loss_G (média móvel)')
        ax4.set_title('Análise de Convergência (Média Móvel)', fontweight='bold')
        ax4.set_xlabel('Época')
        ax4.set_ylabel('Loss')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
    else:
        ax4.plot(epochs, loss_d, 'b-', linewidth=2, label='Loss_D')
        ax4.plot(epochs, loss_g, 'r-', linewidth=2, label='Loss_G')
        ax4.set_title('Análise de Convergência', fontweight='bold')
        ax4.set_xlabel('Época')
        ax4.set_ylabel('Loss')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
    
    plt.tight_layout()
    
    # Salvar gráfico
    plot_path = os.path.join(output_dir, 'training_evolution.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Gráfico atualizado: {plot_path}")

def update_training_logs(out_dir, epoch, loss_d, loss_g, loss_g_gan, loss_g_l1, epoch_time):
    """Atualiza os logs de treinamento com componentes separados do Loss_G"""
    log_file = os.path.join(out_dir, 'training_log.txt')
    
    # Formato do log com componentes separados
    log_line = f"Época {epoch} - Loss_D: {loss_d:.4f}, Loss_G: {loss_g:.4f} (GAN: {loss_g_gan:.4f}, L1: {loss_g_l1:.4f}) (Tempo: {epoch_time:.2f}s)\n"
    
    # Salvar no arquivo de log
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(log_line)
    
    print(f"📝 Log atualizado: {log_line.strip()}")
    
    # Atualizar gráficos
    try:
        data = parse_training_logs(log_file)
        if data:
            print(f"📊 Gerando gráfico para época {epoch}...")
            success = create_training_plots(data, out_dir)
            if success:
                print(f"✅ Gráfico atualizado para época {epoch}")
            else:
                print(f"❌ Falha ao gerar gráfico para época {epoch}")
        else:
            print(f"⚠️ Dados não encontrados para gerar gráfico na época {epoch}")
    except Exception as e:
        print(f"❌ Erro ao gerar gráfico na época {epoch}: {e}")

def real_train(config, max_images=None, pretrained_model=None):
    """Treinamento com dados reais com early stopping e learning rate scheduling"""
    
    print("🚀 Iniciando treinamento com dados reais...")
    
    # Fazer backup dos dados anteriores
    if not backup_previous_results(config.out_dir):
        print("❌ Falha no backup - abortando treinamento")
        return False
    
    # Configurar GPU
    if config.cuda:
        torch.cuda.set_device(config.gpu_ids[0])
    
    # Carregar dataset real
    train_A_dir = os.path.join(config.datasets_dir, 'train_A')
    train_C_dir = os.path.join(config.datasets_dir, 'train_C')
    
    if not os.path.exists(train_A_dir) or not os.path.exists(train_C_dir):
        print(f"❌ Diretórios de dataset não encontrados:")
        print(f"   {train_A_dir}")
        print(f"   {train_C_dir}")
        return False
    
    dataset = RealDataset(train_A_dir, train_C_dir, config.width, config.height, max_images)
    dataloader = DataLoader(dataset, batch_size=config.batchsize, shuffle=True, num_workers=0)
    
    # Carregar modelos
    print("📦 Carregando modelos...")
    gen = Generator(gpu_ids=config.gpu_ids)
    dis = Discriminator(in_ch=config.in_ch, out_ch=config.out_ch, gpu_ids=config.gpu_ids)
    
    # Carregar modelo pré-treinado se especificado
    if pretrained_model and os.path.exists(pretrained_model):
        print(f"🔄 Carregando modelo pré-treinado: {pretrained_model}")
        try:
            gen.load_state_dict(torch.load(pretrained_model, map_location='cpu'))
            print("✅ Modelo pré-treinado carregado com sucesso!")
        except Exception as e:
            print(f"❌ Erro ao carregar modelo pré-treinado: {e}")
            print("🔄 Continuando com modelo inicializado aleatoriamente...")
    else:
        print("🔄 Iniciando com modelo inicializado aleatoriamente...")
    
    if config.cuda:
        gen = gen.cuda()
        dis = dis.cuda()
    
    # Otimizadores
    opt_gen = optim.Adam(gen.parameters(), lr=config.lr, betas=(config.beta1, 0.999))
    opt_dis = optim.Adam(dis.parameters(), lr=config.lr, betas=(config.beta1, 0.999))
    
    # Critérios
    criterionL1 = nn.L1Loss()
    criterionSoftplus = nn.Softplus()
    
    if config.cuda:
        criterionL1 = criterionL1.cuda()
        criterionSoftplus = criterionSoftplus.cuda()
    
    # Criar diretório de saída
    os.makedirs(config.out_dir, exist_ok=True)
    
    # Criar arquivo de log
    log_file = os.path.join(config.out_dir, 'training_log.txt')
    
    # Cabeçalho do log
    header = f"🚀 Treinamento com dados reais - Iniciado em {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    header += f"📊 Dataset: {len(dataset)} imagens\n"
    header += f"📊 Épocas: {config.epoch}\n"
    header += f"📊 Batch size: {config.batchsize}\n"
    header += "="*50 + "\n"
    
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(header)
    
    print("🎯 Iniciando loop de treinamento...")
    print(f"📊 Épocas: {config.epoch}")
    print(f"📊 Batch size: {config.batchsize}")
    print(f"📊 Imagens por época: {len(dataset)}")
    
    # Variáveis para early stopping e learning rate scheduling
    best_loss_g_l1 = float('inf')
    patience_counter = 0
    patience_limit = 5  # Número de épocas para aguardar antes de ajustar
    lr_reduction_factor = 0.5
    min_lr = 1e-6
    checkpoint_backup = None
    
    # Loop de treinamento
    for epoch in range(1, config.epoch + 1):
        epoch_start_time = time.time()
        epoch_loss_d = 0.0
        epoch_loss_g = 0.0
        epoch_loss_g_gan = 0.0
        epoch_loss_g_l1 = 0.0
        
        print(f"🔄 Época {epoch}/{config.epoch}")
        
        for batch_idx, (real_a, real_b) in enumerate(dataloader):
            if config.cuda:
                real_a = real_a.cuda()
                real_b = real_b.cuda()
            
            real_a = Variable(real_a)
            real_b = Variable(real_b)
            
            # Forward pass do gerador
            att, fake_b = gen.forward(real_a)
            
            # Treinar discriminador
            opt_dis.zero_grad()
            
            fake_ab = torch.cat((real_a, fake_b), 1)
            pred_fake = dis.forward(fake_ab.detach())
            loss_d_fake = criterionSoftplus(pred_fake).mean()
            
            real_ab = torch.cat((real_a, real_b), 1)
            pred_real = dis.forward(real_ab)
            loss_d_real = criterionSoftplus(-pred_real).mean()
            
            loss_d = (loss_d_real + loss_d_fake) * 0.5
            loss_d.backward()
            opt_dis.step()
            
            # Treinar gerador
            opt_gen.zero_grad()
            
            fake_ab = torch.cat((real_a, fake_b), 1)
            pred_fake = dis.forward(fake_ab)
            loss_g_gan = criterionSoftplus(-pred_fake).mean()
            
            loss_g_l1 = criterionL1(fake_b, real_b) * config.lamb
            loss_g = loss_g_gan + loss_g_l1
            
            loss_g.backward()
            opt_gen.step()
            
            epoch_loss_d += loss_d.item()
            epoch_loss_g += loss_g.item()
            epoch_loss_g_gan += loss_g_gan.item()
            epoch_loss_g_l1 += loss_g_l1.item()
            
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}/{len(dataloader)}: Loss_D={loss_d.item():.4f}, Loss_G={loss_g.item():.4f} (GAN: {loss_g_gan.item():.4f}, L1: {loss_g_l1.item():.4f})")
        
        # Calcular médias da época
        avg_loss_d = epoch_loss_d / len(dataloader)
        avg_loss_g = epoch_loss_g / len(dataloader)
        avg_loss_g_gan = epoch_loss_g_gan / len(dataloader)
        avg_loss_g_l1 = epoch_loss_g_l1 / len(dataloader)
        
        epoch_time = time.time() - epoch_start_time
        
        print(f"  Época {epoch} - Loss_D: {avg_loss_d:.4f}, Loss_G: {avg_loss_g:.4f} (GAN: {avg_loss_g_gan:.4f}, L1: {avg_loss_g_l1:.4f})")
        
        # Early stopping e learning rate scheduling baseado no Loss_G_L1
        if avg_loss_g_l1 < best_loss_g_l1:
            best_loss_g_l1 = avg_loss_g_l1
            patience_counter = 0
            print(f"✅ Novo melhor Loss_G_L1: {best_loss_g_l1:.4f}")
            
            # Fazer backup do melhor modelo
            if checkpoint_backup is not None:
                # Restaurar backup anterior
                gen.load_state_dict(checkpoint_backup['gen'])
                dis.load_state_dict(checkpoint_backup['dis'])
                opt_gen.load_state_dict(checkpoint_backup['opt_gen'])
                opt_dis.load_state_dict(checkpoint_backup['opt_dis'])
                print(f"🔄 Restaurado backup da época {checkpoint_backup['epoch']}")
        else:
            patience_counter += 1
            print(f"⚠️ Loss_G_L1 aumentou: {avg_loss_g_l1:.4f} > {best_loss_g_l1:.4f} (patience: {patience_counter}/{patience_limit})")
            
            # Fazer backup do estado atual antes de ajustar
            checkpoint_backup = {
                'gen': gen.state_dict().copy(),
                'dis': dis.state_dict().copy(),
                'opt_gen': opt_gen.state_dict().copy(),
                'opt_dis': opt_dis.state_dict().copy(),
                'epoch': epoch
            }
            
            # Ajustar learning rate se necessário
            if patience_counter >= patience_limit:
                current_lr = opt_gen.param_groups[0]['lr']
                new_lr = current_lr * lr_reduction_factor
                
                if new_lr >= min_lr:
                    for param_group in opt_gen.param_groups:
                        param_group['lr'] = new_lr
                    for param_group in opt_dis.param_groups:
                        param_group['lr'] = new_lr
                    
                    print(f"🔄 Learning rate reduzido: {current_lr:.6f} → {new_lr:.6f}")
                    patience_counter = 0  # Resetar contador
                else:
                    print(f"🛑 Learning rate mínimo atingido ({min_lr:.6f}). Parando treinamento.")
                    break
        
        # Atualizar logs e gráficos
        update_training_logs(config.out_dir, epoch, avg_loss_d, avg_loss_g, avg_loss_g_gan, avg_loss_g_l1, epoch_time)
        
        # Salvar checkpoint
        if epoch % config.snapshot_interval == 0:
            simple_checkpoint(gen, epoch, config.out_dir, 'gen')
            simple_checkpoint(dis, epoch, config.out_dir, 'dis')
            print(f"💾 Checkpoint salvo na época {epoch}")
        
        print(f"✅ Época {epoch} concluída em {epoch_time:.2f}s")
    
    # Finalizar log
    footer = f"\n🎉 Treinamento concluído em {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(footer)
    
    print("🎉 Treinamento concluído!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Treinamento com dados reais do SpA-Former')
    parser.add_argument('--config', type=str, required=True, help='Caminho para o arquivo de configuração')
    parser.add_argument('--max_images', type=int, default=None, help='Número máximo de imagens para treinar (padrão: todas)')
    parser.add_argument('--pretrained_model', type=str, default=None, help='Caminho para modelo pré-treinado')
    args = parser.parse_args()
    
    with open(args.config, 'r', encoding='UTF-8') as f:
        config_dict = yaml.safe_load(f)
    
    config = SimpleConfig(config_dict)
    
    print("🚀 Treinamento com dados reais do SpA-Former")
    print(f"📂 Configuração: {args.config}")
    print(f"📊 Dataset: {config.datasets_dir}")
    print(f"📊 Validação: {config.valset_dir}")
    print(f"📊 Épocas: {config.epoch}")
    print(f"📊 Batch size: {config.batchsize}")
    print(f"📊 Imagens: {config.n_data}")
    if args.max_images:
        print(f"📊 Imagens limitadas: {args.max_images}")
    if args.pretrained_model:
        print(f"🤖 Modelo pré-treinado: {args.pretrained_model}")
    print(f"💾 Saída: {config.out_dir}")
    
    real_train(config, max_images=args.max_images, pretrained_model=args.pretrained_model) 