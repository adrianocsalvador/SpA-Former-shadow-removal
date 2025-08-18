#!/usr/bin/env python3
"""
Predição otimizada para imagens panorâmicas grandes com reconstrução perfeita
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

from SpA_Former import Generator
from utils import gpu_manage, save_image, heatmap


class PanoramicPredictor:
    """Preditor otimizado para imagens panorâmicas grandes"""
    
    def __init__(self, model_path, config, window_size=(1024, 1024), stride=(512, 512)):
        self.config = config
        self.window_size = window_size
        self.stride = stride
        
        # Carregar modelo
        self.gen = Generator(gpu_ids=config.gpu_ids)
        param = torch.load(model_path)
        self.gen.load_state_dict(param)
        
        if config.cuda:
            self.gen = self.gen.cuda()
        
        self.gen.eval()
    
    def process_image(self, image_path, output_path):
        """Processa uma imagem panorâmica grande"""
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
        
        # Processar cada janela
        with torch.no_grad():
            for i, (x, y) in enumerate(tqdm(windows, desc="Processando janelas")):
                # Extrair janela
                window = img_normalized[y:y+self.window_size[1], x:x+self.window_size[0]]
                
                # Converter para tensor
                window_tensor = torch.from_numpy(window.transpose(2, 0, 1)).unsqueeze(0)
                
                if self.config.cuda:
                    window_tensor = window_tensor.cuda()
                
                # Predição
                if hasattr(self.config, 'use_mixed_precision') and self.config.use_mixed_precision:
                    with autocast():
                        att, output_window = self.gen(window_tensor)
                else:
                    att, output_window = self.gen(window_tensor)
                
                # Converter de volta para numpy
                output_window = output_window.cpu().numpy()[0].transpose(1, 2, 0)
                output_window = np.clip(output_window, 0, 1)
                
                # Aplicar janela de suavização (blending)
                blend_window = self._create_blend_window(self.window_size[0], self.window_size[1])
                
                # Adicionar à imagem de saída com blending
                output_img[y:y+self.window_size[1], x:x+self.window_size[0]] += output_window * blend_window[:, :, np.newaxis]
                count_map[y:y+self.window_size[1], x:x+self.window_size[0]] += blend_window
        
        # Normalizar pela contagem
        count_map[count_map == 0] = 1  # Evitar divisão por zero
        output_img = output_img / count_map[:, :, np.newaxis]
        
        # Converter de volta para uint8
        output_img = np.clip(output_img * 255, 0, 255).astype(np.uint8)
        
        # Salvar resultado
        cv2.imwrite(output_path, output_img)
        print(f"Resultado salvo em: {output_path}")
        
        return output_img
    
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
        """Cria janela de suavização para blending"""
        # Criar janela de suavização (feather)
        blend = np.ones((h, w), dtype=np.float32)
        
        # Aplicar suavização nas bordas
        feather_size = min(w, h) // 8
        
        for i in range(feather_size):
            # Borda superior
            blend[i, :] = i / feather_size
            # Borda inferior
            blend[h-1-i, :] = i / feather_size
            # Borda esquerda
            blend[:, i] = np.minimum(blend[:, i], i / feather_size)
            # Borda direita
            blend[:, w-1-i] = np.minimum(blend[:, w-1-i], i / feather_size)
        
        return blend


def predict_panoramic(config, args):
    """Função principal de predição para panorâmicas"""
    gpu_manage(args)
    
    # Criar preditor
    predictor = PanoramicPredictor(
        model_path=args.pretrained,
        config=config,
        window_size=(args.window_size, args.window_size),
        stride=(args.stride, args.stride)
    )
    
    # Processar imagem única ou diretório
    if os.path.isfile(args.test_dir):
        # Arquivo único
        output_path = os.path.join(args.out_dir, f"output_{os.path.basename(args.test_dir)}")
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
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Converter para objeto simples
    class SimpleConfig:
        def __init__(self, config_dict):
            for key, value in config_dict.items():
                setattr(self, key, value)
    
    config = SimpleConfig(config)
    predict_panoramic(config, args)
