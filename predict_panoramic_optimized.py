#!/usr/bin/env python3
"""
Predição otimizada para imagens panorâmicas com gerenciamento de memória avançado
"""

import numpy as np
import argparse
from tqdm import tqdm
import yaml
import cv2
import os
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
import gc

from SpA_Former import Generator
from utils import gpu_manage, save_image, heatmap


class PanoramicPredictorOptimized:
    """Preditor otimizado para imagens panorâmicas grandes com gerenciamento de memória"""
    
    def __init__(self, model_path, config, window_size=(1024, 1024), stride=(512, 512),
                 blend_method='hann', blend_sigma_frac=0.25, blend_min_weight=1e-3,
                 exposure_comp=False, exposure_alpha=0.5):
        self.config = config
        self.window_size = window_size
        self.stride = stride
        self.blend_method = blend_method
        self.blend_sigma_frac = float(blend_sigma_frac)
        self.blend_min_weight = float(blend_min_weight)
        self.exposure_comp = bool(exposure_comp)
        self.exposure_alpha = float(exposure_alpha)
        
        # Configurar PyTorch para melhor gerenciamento de memória
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        # Carregar modelo
        print("Carregando modelo...")
        self.gen = Generator(gpu_ids=[0] if config.cuda else [])
        param = torch.load(model_path, map_location='cpu')
        self.gen.load_state_dict(param)
        
        if config.cuda:
            self.gen = self.gen.cuda()
        
        self.gen.eval()
        print("Modelo carregado com sucesso!")
    
    def process_image(self, image_path, output_path):
        """Processa uma imagem panorâmica grande com otimizações de memória"""
        print(f"Processando: {image_path}")
        
        # Carregar imagem
        img = cv2.imread(image_path, 1).astype(np.float32)
        if img is None:
            print(f"Erro ao carregar imagem: {image_path}")
            return
        
        h, w = img.shape[:2]
        print(f"Tamanho da imagem: {w}x{h}")
        
        # Normalizar
        img_normalized = img / 255.0
        
        # Criar canvas de saída
        output_img = np.zeros_like(img_normalized)
        count_map = np.zeros((h, w), dtype=np.float32)
        
        # Calcular janelas
        windows = self._generate_windows(h, w)
        print(f"Total de janelas: {len(windows)}")
        
        # Processar cada janela com gerenciamento de memória
        with torch.no_grad():
            for i, (x, y) in enumerate(tqdm(windows, desc="Processando janelas")):
                # Limpar cache a cada 10 janelas
                if i % 10 == 0:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                
                # Extrair janela
                window = img_normalized[y:y+self.window_size[1], x:x+self.window_size[0]]
                
                # Converter para tensor
                window_tensor = torch.from_numpy(window.transpose(2, 0, 1)).unsqueeze(0)
                
                if self.config.cuda:
                    window_tensor = window_tensor.cuda()
                
                # Predição com mixed precision se habilitado
                try:
                    if hasattr(self.config, 'use_mixed_precision') and self.config.use_mixed_precision:
                        # Usar versão mais nova do autocast
                        with torch.amp.autocast('cuda'):
                            att, output_window = self.gen(window_tensor)
                    else:
                        att, output_window = self.gen(window_tensor)
                    
                    # Converter de volta para numpy
                    output_window = output_window.detach().cpu().numpy()[0].transpose(1, 2, 0)
                    
                    # Verificar e corrigir valores inválidos
                    if np.any(np.isnan(output_window)) or np.any(np.isinf(output_window)):
                        print(f"⚠️ Valores inválidos detectados na janela {i}, substituindo por zeros")
                        output_window = np.nan_to_num(output_window, nan=0.0, posinf=1.0, neginf=0.0)
                    
                    output_window = np.clip(output_window, 0, 1)
                    
                    # Aplicar compensação de exposição no overlap (opcional)
                    if self.exposure_comp:
                        overlap_mask = count_map[y:y+self.window_size[1], x:x+self.window_size[0]] > 0
                        if np.any(overlap_mask):
                            existing = output_img[y:y+self.window_size[1], x:x+self.window_size[0]].copy()
                            # médias por canal na região de overlap
                            for ch in range(3):
                                mean_existing = existing[:, :, ch][overlap_mask].mean() if overlap_mask.any() else 0.0
                                mean_patch = output_window[:, :, ch][overlap_mask].mean() if overlap_mask.any() else 0.0
                                delta = mean_existing - mean_patch
                                output_window[:, :, ch] = np.clip(output_window[:, :, ch] + self.exposure_alpha * delta, 0.0, 1.0)

                    # Aplicar janela de suavização (blending)
                    blend_window = self._create_blend_window(self.window_size[0], self.window_size[1])
                    
                    # Adicionar à imagem de saída com blending
                    output_img[y:y+self.window_size[1], x:x+self.window_size[0]] += output_window * blend_window[:, :, np.newaxis]
                    count_map[y:y+self.window_size[1], x:x+self.window_size[0]] += blend_window
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"\n❌ Erro de memória na janela {i}. Reduzindo tamanho...")
                        # Tentar com janela menor
                        smaller_window = self._process_with_smaller_window(window, x, y, output_img, count_map)
                        if not smaller_window:
                            print(f"❌ Falha mesmo com janela menor. Pulando janela {i}")
                            continue
                    else:
                        print(f"\n❌ Erro na janela {i}: {e}")
                        continue
                
                # Liberar memória
                del window_tensor, output_window
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Normalizar pela contagem
        count_map[count_map == 0] = 1  # Evitar divisão por zero
        output_img = output_img / count_map[:, :, np.newaxis]
        
        # Verificar e corrigir valores finais
        if np.any(np.isnan(output_img)) or np.any(np.isinf(output_img)):
            print("⚠️ Valores inválidos na imagem final, corrigindo...")
            output_img = np.nan_to_num(output_img, nan=0.0, posinf=1.0, neginf=0.0)
        
        # Converter de volta para uint8
        output_img = np.clip(output_img * 255, 0, 255).astype(np.uint8)
        
        # Salvar resultado
        cv2.imwrite(output_path, output_img)
        print(f"Resultado salvo em: {output_path}")
        
        return output_img
    
    def _process_with_smaller_window(self, window, x, y, output_img, count_map):
        """Processa com janela menor em caso de erro de memória"""
        try:
            # Reduzir tamanho da janela pela metade
            h, w = window.shape[:2]
            new_h, new_w = h // 2, w // 2
            
            # Processar em 4 quadrantes
            quadrants = [
                (0, 0, new_w, new_h),
                (new_w, 0, w, new_h),
                (0, new_h, new_w, h),
                (new_w, new_h, w, h)
            ]
            
            for qx, qy, qw, qh in quadrants:
                sub_window = window[qy:qh, qx:qw]
                if sub_window.size == 0:
                    continue
                
                # Converter para tensor
                sub_tensor = torch.from_numpy(sub_window.transpose(2, 0, 1)).unsqueeze(0)
                if self.config.cuda:
                    sub_tensor = sub_tensor.cuda()
                
                # Predição
                if hasattr(self.config, 'use_mixed_precision') and self.config.use_mixed_precision:
                    with autocast():
                        att, output_sub = self.gen(sub_tensor)
                else:
                    att, output_sub = self.gen(sub_tensor)
                
                # Converter de volta
                output_sub = output_sub.cpu().numpy()[0].transpose(1, 2, 0)
                output_sub = np.clip(output_sub, 0, 1)
                
                # Aplicar à posição correta
                output_img[y+qy:y+qh, x+qx:x+qw] += output_sub
                count_map[y+qy:y+qh, x+qx:x+qw] += 1
                
                del sub_tensor, output_sub
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            return True
            
        except Exception as e:
            print(f"Erro mesmo com janela menor: {e}")
            return False
    
    def _generate_windows(self, h, w):
        """Gera coordenadas das janelas com overlap"""
        windows = []
        
        # Janelas principais
        for y in range(0, h - self.window_size[1] + 1, self.stride[1]):
            for x in range(0, w - self.window_size[0] + 1, self.stride[0]):
                windows.append((x, y))
        
        # Janelas finais para cobrir bordas
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
        
        return list(set(windows))
    
    def _create_blend_window(self, w, h):
        """Cria janela de suavização para blending (hann/cosine, gaussian ou linear)."""
        method = str(self.blend_method).lower()
        if method in ('hann', 'hanning', 'cosine', 'cos'):
            wx = np.hanning(w).astype(np.float32)
            wy = np.hanning(h).astype(np.float32)
            blend = np.outer(wy, wx)
            # Normalizar para máximo = 1 e evitar zeros absolutos
            if blend.max() > 0:
                blend = blend / blend.max()
            blend = np.clip(blend, self.blend_min_weight, 1.0)
            return blend.astype(np.float32)
        elif method in ('gaussian', 'gauss'):
            yy, xx = np.mgrid[0:h, 0:w]
            cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
            sigma = self.blend_sigma_frac * min(w, h)
            sigma = max(sigma, 1.0)
            d2 = ((yy - cy) ** 2 + (xx - cx) ** 2)
            blend = np.exp(-0.5 * d2 / (sigma ** 2)).astype(np.float32)
            if blend.max() > 0:
                blend = blend / blend.max()
            blend = np.clip(blend, self.blend_min_weight, 1.0)
            return blend
        else:
            # Linear feather nas bordas (compatível com versão anterior)
            blend = np.ones((h, w), dtype=np.float32)
            feather_size = max(min(w, h) // 8, 1)
            # Ramp vertical
            ramp_y = np.linspace(0, 1, feather_size, dtype=np.float32)
            blend[:feather_size, :] = ramp_y[:, None]
            blend[h - feather_size:, :] = ramp_y[::-1][:, None]
            # Ramp horizontal
            ramp_x = np.linspace(0, 1, feather_size, dtype=np.float32)
            blend[:, :feather_size] = np.minimum(blend[:, :feather_size], ramp_x[None, :])
            blend[:, w - feather_size:] = np.minimum(blend[:, w - feather_size:], ramp_x[::-1][None, :])
            blend = np.clip(blend, self.blend_min_weight, 1.0)
            return blend


def predict_panoramic_optimized(config, args):
    """Função principal de predição otimizada para panorâmicas"""
    # Configurar GPU manualmente para evitar problemas com gpu_manage
    if args.cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
            print(f"Usando GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("CUDA não disponível, usando CPU")
            args.cuda = False
    else:
        print("Usando CPU")
    
    # Configurar seed
    torch.manual_seed(42)
    if args.cuda:
        torch.cuda.manual_seed_all(42)
    
    # Configurar variável de ambiente para melhor gerenciamento de memória
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # Criar preditor
    predictor = PanoramicPredictorOptimized(
        model_path=args.pretrained,
        config=config,
        window_size=(args.window_size, args.window_size),
        stride=(args.stride, args.stride),
        blend_method=args.blend_method,
        blend_sigma_frac=args.blend_sigma_frac,
        blend_min_weight=args.blend_min_weight,
        exposure_comp=args.exposure_comp,
        exposure_alpha=args.exposure_alpha
    )
    
    # Processar imagem única ou diretório
    if os.path.isfile(args.test_dir):
        # Arquivo único
        output_path = os.path.join(args.out_dir, f"output_{os.path.basename(args.test_dir)}")
        os.makedirs(args.out_dir, exist_ok=True)
        predictor.process_image(args.test_dir, output_path)
    else:
        # Diretório
        if not os.path.exists(args.out_dir):
            os.makedirs(args.out_dir)
        
        image_extensions = ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']
        image_files = [f for f in os.listdir(args.test_dir) 
                      if any(f.lower().endswith(ext) for ext in image_extensions)]
        
        for image_file in tqdm(image_files, desc="Processando imagens"):
            input_path = os.path.join(args.test_dir, image_file)
            output_path = os.path.join(args.out_dir, f"output_{image_file}")
            predictor.process_image(input_path, output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config_panoramic.yml', help='path to config file')
    parser.add_argument('--test_dir', type=str, required=True, help='path to test image or directory')
    parser.add_argument('--out_dir', type=str, required=True, help='path to output directory')
    parser.add_argument('--pretrained', type=str, required=True, help='path to pretrained model')
    parser.add_argument('--cuda', action='store_true', help='use cuda')
    parser.add_argument('--window_size', type=int, default=1024, help='window size for processing')
    parser.add_argument('--stride', type=int, default=512, help='stride between windows')
    parser.add_argument('--blend_method', type=str, default='hann', choices=['hann','cosine','gaussian','linear'], help='blending window method')
    parser.add_argument('--blend_sigma_frac', type=float, default=0.25, help='sigma fraction for gaussian blending (relative to min(window))')
    parser.add_argument('--blend_min_weight', type=float, default=1e-3, help='minimum weight to avoid hard zeros at edges')
    parser.add_argument('--exposure_comp', action='store_true', help='enable exposure compensation over overlaps')
    parser.add_argument('--exposure_alpha', type=float, default=0.5, help='exposure compensation strength [0-1]')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Converter para objeto simples
    class SimpleConfig:
        def __init__(self, config_dict):
            for key, value in config_dict.items():
                setattr(self, key, value)
    
    config = SimpleConfig(config)
    predict_panoramic_optimized(config, args)
