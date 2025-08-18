#!/usr/bin/env python3
"""
Treinamento com logs detalhados, gráficos e timestamp
Versão adaptada para janelas 512x512 com otimizações ULTRA de memória
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
import datetime
import json
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except Exception:
    MATPLOTLIB_AVAILABLE = False
from pathlib import Path

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


def setup_memory_optimization():
    """Configura otimizações ULTRA de memória"""
    # Configurar PyTorch para melhor gerenciamento de memória
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:128'
    
    # Limpar cache inicial
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    
    # Configurar para usar menos memória
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def clear_memory_aggressive():
    """Limpeza ULTRA agressiva de memória"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    gc.collect()


class TrainingLogger:
    """Logger para treinamento com gráficos e logs tabulares"""
    
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.log_file = os.path.join(output_dir, "training_log.txt")
        self.metrics_file = os.path.join(output_dir, "metrics.json")
        self.enable_plots = MATPLOTLIB_AVAILABLE
        self.metrics = {
            'epochs': [],
            'loss_d': [],
            'loss_g': [],
            'loss_l1': [],
            'loss_whitening': [],
            'epoch_time': [],
            'total_time': [],
            'memory_usage': []
        }
        
        # Criar diretório se não existir
        os.makedirs(output_dir, exist_ok=True)
        
        # Inicializar arquivo de log
        with open(self.log_file, 'w') as f:
            f.write("=== TREINAMENTO SPA-FORMER 512x512 ULTRA ===\n")
            f.write(f"Início: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 50 + "\n\n")
            if not self.enable_plots:
                f.write("Aviso: gráficos desabilitados (Matplotlib indisponível).\n\n")
    
    def log_epoch(self, epoch, total_epochs, loss_d, loss_g, loss_l1, loss_whitening, epoch_time, total_time, memory_usage):
        """Log de uma época"""
        # Adicionar métricas
        self.metrics['epochs'].append(epoch)
        self.metrics['loss_d'].append(float(loss_d))
        self.metrics['loss_g'].append(float(loss_g))
        self.metrics['loss_l1'].append(float(loss_l1))
        self.metrics['loss_whitening'].append(float(loss_whitening))
        self.metrics['epoch_time'].append(epoch_time)
        self.metrics['total_time'].append(total_time)
        self.metrics['memory_usage'].append(memory_usage)
        
        # Log para arquivo
        with open(self.log_file, 'a') as f:
            f.write(f"Época {epoch:3d}/{total_epochs:3d} | ")
            f.write(f"Loss_D: {loss_d:.4f} | ")
            f.write(f"Loss_G: {loss_g:.4f} | ")
            f.write(f"L1: {loss_l1:.4f} | ")
            f.write(f"Whitening: {loss_whitening:.4f} | ")
            f.write(f"Tempo Época: {epoch_time:.2f}s | ")
            f.write(f"Tempo Total: {total_time:.2f}s | ")
            f.write(f"Memória: {memory_usage:.1f}GB\n")
        
        # Salvar métricas em JSON
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        # Gerar gráficos
        self.plot_metrics()
    
    def log_iteration(self, epoch, iteration, total_iterations, loss_d, loss_g, loss_l1, loss_whitening):
        """Log de uma iteração"""
        with open(self.log_file, 'a') as f:
            f.write(f"  Iter {iteration:4d}/{total_iterations:4d} | ")
            f.write(f"Loss_D: {loss_d:.4f} | ")
            f.write(f"Loss_G: {loss_g:.4f} | ")
            f.write(f"L1: {loss_l1:.4f} | ")
            f.write(f"Whitening: {loss_whitening:.4f}\n")
    
    def plot_metrics(self):
        """Gera gráficos das métricas"""
        if not self.enable_plots or len(self.metrics['epochs']) < 2:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Métricas de Treinamento SPA-Former 512x512 ULTRA', fontsize=16)
        
        # Gráfico 1: Losses principais
        axes[0, 0].plot(self.metrics['epochs'], self.metrics['loss_d'], 'r-', label='Loss D')
        axes[0, 0].plot(self.metrics['epochs'], self.metrics['loss_g'], 'b-', label='Loss G')
        axes[0, 0].set_title('Loss Discriminator vs Generator')
        axes[0, 0].set_xlabel('Época')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Gráfico 2: Losses detalhadas
        axes[0, 1].plot(self.metrics['epochs'], self.metrics['loss_l1'], 'g-', label='L1 Loss')
        axes[0, 1].plot(self.metrics['epochs'], self.metrics['loss_whitening'], 'm-', label='Whitening Penalty')
        axes[0, 1].set_title('Losses Detalhadas')
        axes[0, 1].set_xlabel('Época')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Gráfico 3: Tempo por época
        axes[1, 0].plot(self.metrics['epochs'], self.metrics['epoch_time'], 'c-', label='Tempo/Época')
        axes[1, 0].set_title('Tempo por Época')
        axes[1, 0].set_xlabel('Época')
        axes[1, 0].set_ylabel('Tempo (s)')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Gráfico 4: Uso de memória
        if self.metrics['memory_usage']:
            axes[1, 1].plot(self.metrics['epochs'], self.metrics['memory_usage'], 'y-', label='Memória GPU')
            axes[1, 1].set_title('Uso de Memória GPU')
            axes[1, 1].set_xlabel('Época')
            axes[1, 1].set_ylabel('Memória (GB)')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'training_metrics.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def log_final(self, total_time, final_metrics):
        """Log final do treinamento"""
        with open(self.log_file, 'a') as f:
            f.write("\n" + "=" * 50 + "\n")
            f.write("TREINAMENTO 512x512 ULTRA CONCLUÍDO\n")
            f.write(f"Tempo Total: {total_time:.2f}s ({total_time/3600:.2f}h)\n")
            f.write(f"Final: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("Métricas Finais:\n")
            for key, value in final_metrics.items():
                f.write(f"  {key}: {value}\n")
            f.write("=" * 50 + "\n")


class MemoryOptimizedDataset512x512:
    """Dataset otimizado para economia máxima de memória - versão 512x512"""
    
    def __init__(self, config, window_size=(512, 512), stride=(256, 256)):
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
                shadow_indices = [i for i, w in enumerate(self.current_windows) if float(w[3]) > 0.1]
                if shadow_indices:
                    idx = random.choice(shadow_indices)
                    window_data = self.current_windows.pop(idx)
                else:
                    window_data = self.current_windows.pop(0)
            else:
                # Segunda metade: janelas sem sombras ou com baixo contraste
                low_contrast_indices = [i for i, w in enumerate(self.current_windows) if float(w[4]) < 30 or float(w[3]) < 0.05]
                if low_contrast_indices:
                    idx = random.choice(low_contrast_indices)
                    window_data = self.current_windows.pop(idx)
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


def get_memory_usage():
    """Obtém uso atual de memória GPU"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3  # GB
    return 0


def train_panoramic_512x512_ultra(config):
    """Função principal de treinamento com logs detalhados - versão 512x512 ULTRA"""
    
    # Setup inicial de memória ULTRA
    setup_memory_optimization()
    gpu_manage(config)

    # Criar diretório com timestamp
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M')
    output_dir = f"./results_512x512_ultra_{timestamp}"
    config.out_dir = output_dir
    
    # Inicializar logger
    logger = TrainingLogger(output_dir)
    
    print(f'📁 Resultados salvos em: {output_dir}')
    print('===> Loading panoramic dataset 512x512 ULTRA')
    
    # Configurar tamanho da janela e stride
    window_size = (getattr(config, 'window_width', 512), getattr(config, 'window_height', 512))
    stride = (getattr(config, 'stride_width', 256), getattr(config, 'stride_height', 256))
    
    train_dataset = MemoryOptimizedDataset512x512(config, window_size, stride)
    validation_dataset = ValDataset(config)
    
    print(f'Panoramic dataset 512x512: {len(train_dataset)} total windows')
    print(f'Window size: {window_size}, Stride: {stride}')
    print('validation dataset:', len(validation_dataset))

    validation_data_loader = DataLoader(dataset=validation_dataset, num_workers=config.threads, 
                                       batch_size=config.validation_batchsize, shuffle=False)
    
    print('===> Loading models')

    gen = Generator(gpu_ids=config.gpu_ids)
    
    # Carregar modelo pré-treinado se especificado
    if hasattr(config, 'gen_init') and config.gen_init:
        try:
            param = torch.load(config.gen_init, weights_only=True)
            gen.load_state_dict(param)
            print('load {} as pretrained model'.format(config.gen_init))
        except Exception as e:
            print(f'⚠️ Erro ao carregar modelo pré-treinado: {e}')
            print('✅ Iniciando treinamento do zero')

    dis = Discriminator(in_ch=config.in_ch, out_ch=config.out_ch, gpu_ids=config.gpu_ids)

    if hasattr(config, 'dis_init') and config.dis_init:
        param = torch.load(config.dis_init, weights_only=True)
        dis.load_state_dict(param)
        print('load {} as pretrained model'.format(config.dis_init))

    # Setup optimizers
    opt_gen = optim.Adam(gen.parameters(), lr=config.lr, betas=(config.beta1, 0.999), weight_decay=0.00001)
    opt_dis = optim.Adam(dis.parameters(), lr=config.lr, betas=(config.beta1, 0.999), weight_decay=0.00001)
    
    # Loss functions
    adaptive_loss = AdaptiveLoss(lambda_l1=config.lamb, lambda_whitening=getattr(config, 'lambda_whitening', 10))
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
    accumulation_steps = getattr(config, 'gradient_accumulation_steps', 16)
    
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
        
        # Métricas da época
        epoch_loss_d = 0
        epoch_loss_g = 0
        epoch_loss_l1 = 0
        epoch_loss_whitening = 0
        batch_count = 0
        
        for iteration in range(total_batches):
            try:
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

                # Acumular métricas
                epoch_loss_d += loss_d.item()
                epoch_loss_g += loss_g.item()
                epoch_loss_l1 += l1_loss.item()
                epoch_loss_whitening += whitening_penalty.item()
                batch_count += 1

                # Log progress
                if iteration % 10 == 0:
                    logger.log_iteration(epoch, iteration, total_batches, loss_d.item(), loss_g.item(), 
                                       l1_loss.item(), whitening_penalty.item())
                
                # Limpeza ULTRA de memória a cada 5 iterações
                if iteration % getattr(config, 'cleanup_frequency', 5) == 0:
                    clear_memory_aggressive()
                    
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"⚠️ OOM na iteração {iteration}, limpando memória ULTRA...")
                    clear_memory_aggressive()
                    continue
                else:
                    raise e

        # Calcular médias da época
        avg_loss_d = epoch_loss_d / batch_count
        avg_loss_g = epoch_loss_g / batch_count
        avg_loss_l1 = epoch_loss_l1 / batch_count
        avg_loss_whitening = epoch_loss_whitening / batch_count
        
        epoch_time = time.time() - epoch_start_time
        total_time = time.time() - start_time
        memory_usage = get_memory_usage()
        
        # Log da época
        logger.log_epoch(epoch, config.epoch, avg_loss_d, avg_loss_g, avg_loss_l1, 
                        avg_loss_whitening, epoch_time, total_time, memory_usage)
        
        print(f'Epoch {epoch} completed in {epoch_time:.2f}s')
        print(f'  Loss_D: {avg_loss_d:.4f}, Loss_G: {avg_loss_g:.4f}')
        print(f'  L1: {avg_loss_l1:.4f}, Whitening: {avg_loss_whitening:.4f}')
        print(f'  Memory: {memory_usage:.1f}GB')

        # Validation e checkpoint a cada época
        if epoch % config.snapshot_interval == 0:
            test(config, validation_data_loader, gen, criterionMSE, epoch)
            checkpoint(config, epoch, gen, dis)
        
        # Limpeza ULTRA de memória após cada época
        clear_memory_aggressive()

    total_time = time.time() - start_time
    
    # Log final
    final_metrics = {
        'final_loss_d': avg_loss_d,
        'final_loss_g': avg_loss_g,
        'final_loss_l1': avg_loss_l1,
        'final_loss_whitening': avg_loss_whitening,
        'total_epochs': config.epoch,
        'total_windows': len(train_dataset),
        'window_size': '512x512',
        'memory_optimization': 'ULTRA'
    }
    logger.log_final(total_time, final_metrics)
    
    print(f'Training 512x512 ULTRA completed in {total_time:.2f}s')
    print(f'📁 Resultados salvos em: {output_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config_training_v9_pairs_512x512_ultra_memory.yml', help='path to config file')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Converter para objeto simples
    class SimpleConfig:
        def __init__(self, config_dict):
            for key, value in config_dict.items():
                setattr(self, key, value)
    
    config = SimpleConfig(config)
    train_panoramic_512x512_ultra(config)

