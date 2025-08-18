#!/usr/bin/env python3
"""
Predição avançada para imagens panorâmicas com blending otimizado
"""

import numpy as np
import argparse
from tqdm import tqdm
import cv2
import os
import torch
import gc

from SpA_Former import Generator


class PanoramicPredictorAdvanced:
    """Preditor avançado para imagens panorâmicas com blending otimizado"""
    
    def __init__(self, model_path, window_size=(1024, 1024), stride=(512, 512), blend_type='gaussian'):
        self.window_size = window_size
        self.stride = stride
        self.blend_type = blend_type
        
        # Configurar PyTorch
        torch.backends.cudnn.benchmark = True
        
        # Carregar modelo
        print("Carregando modelo...")
        self.gen = Generator(gpu_ids=[0])
        param = torch.load(model_path, map_location='cpu', weights_only=True)
        self.gen.load_state_dict(param)
        
        if torch.cuda.is_available():
            self.gen = self.gen.cuda()
        
        self.gen.eval()
        print("Modelo carregado com sucesso!")
    
    def process_image(self, image_path, output_path):
        """Processa uma imagem panorâmica com blending avançado"""
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
                # Limpar cache a cada 20 janelas
                if i % 20 == 0:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                
                # Extrair janela com verificação de limites
                end_y = min(y + self.window_size[1], h)
                end_x = min(x + self.window_size[0], w)
                
                # Verificar se a janela está dentro dos limites
                if end_y <= y or end_x <= x:
                    continue
                
                window = img_normalized[y:end_y, x:end_x]
                
                # Se a janela for menor que o esperado, ajustar o blending
                actual_h, actual_w = window.shape[:2]
                if actual_h != self.window_size[1] or actual_w != self.window_size[0]:
                    # Redimensionar para o tamanho esperado
                    window = cv2.resize(window, (self.window_size[0], self.window_size[1]))
                
                # Converter para tensor
                window_tensor = torch.from_numpy(window.transpose(2, 0, 1)).unsqueeze(0)
                
                if torch.cuda.is_available():
                    window_tensor = window_tensor.cuda()
                
                # Predição
                try:
                    att, output_window = self.gen(window_tensor)
                    
                    # Converter de volta para numpy
                    output_window = output_window.cpu().numpy()[0].transpose(1, 2, 0)
                    
                    # Verificar valores inválidos
                    if np.any(np.isnan(output_window)) or np.any(np.isinf(output_window)):
                        print(f"⚠️ Valores inválidos na janela {i}, usando imagem original")
                        output_window = window
                    
                    output_window = np.clip(output_window, 0, 1)
                    
                    # Aplicar blending avançado
                    blend_window = self._create_advanced_blend(self.window_size[0], self.window_size[1])
                    
                    # Calcular limites reais para aplicar o resultado
                    actual_end_y = min(y + self.window_size[1], h)
                    actual_end_x = min(x + self.window_size[0], w)
                    
                    # Ajustar blending para o tamanho real da janela
                    if actual_end_y - y != self.window_size[1] or actual_end_x - x != self.window_size[0]:
                        # Redimensionar blending para o tamanho real
                        blend_window = cv2.resize(blend_window, (actual_end_x - x, actual_end_y - y))
                        output_window = cv2.resize(output_window, (actual_end_x - x, actual_end_y - y))
                    
                    # Adicionar à imagem de saída com limites corretos
                    output_img[y:actual_end_y, x:actual_end_x] += output_window * blend_window[:, :, np.newaxis]
                    count_map[y:actual_end_y, x:actual_end_x] += blend_window
                    
                except Exception as e:
                    print(f"❌ Erro na janela {i}: {e}")
                    # Usar imagem original para esta janela
                    output_img[y:y+self.window_size[1], x:x+self.window_size[0]] += window
                    count_map[y:y+self.window_size[1], x:x+self.window_size[0]] += 1
                    continue
                
                # Liberar memória
                del window_tensor, output_window
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Normalizar pela contagem
        count_map[count_map == 0] = 1
        output_img = output_img / count_map[:, :, np.newaxis]
        
        # Verificar valores finais
        if np.any(np.isnan(output_img)) or np.any(np.isinf(output_img)):
            print("⚠️ Valores inválidos na imagem final, usando imagem original")
            output_img = img_normalized
        
        # Converter de volta para uint8
        output_img = np.clip(output_img * 255, 0, 255).astype(np.uint8)
        
        # Salvar resultado
        cv2.imwrite(output_path, output_img)
        print(f"Resultado salvo em: {output_path}")
        
        return output_img
    
    def _create_advanced_blend(self, w, h):
        """Cria janela de suavização avançada com diferentes opções"""
        
        if self.blend_type == 'gaussian':
            return self._create_gaussian_blend(w, h)
        elif self.blend_type == 'cosine':
            return self._create_cosine_blend(w, h)
        elif self.blend_type == 'linear':
            return self._create_linear_blend(w, h)
        elif self.blend_type == 'exponential':
            return self._create_exponential_blend(w, h)
        elif self.blend_type == 'radial':
            return self._create_radial_blend(w, h)
        else:
            return self._create_gaussian_blend(w, h)  # Padrão
    
    def _create_gaussian_blend(self, w, h):
        """Blending gaussiano - muito suave no centro"""
        # Criar coordenadas normalizadas
        y_coords, x_coords = np.meshgrid(
            np.linspace(-1, 1, h),
            np.linspace(-1, 1, w),
            indexing='ij'
        )
        
        # Calcular distância do centro
        distance = np.sqrt(x_coords**2 + y_coords**2)
        
        # Aplicar função gaussiana
        sigma = 0.5  # Controle da suavidade
        weight = np.exp(-(distance**2) / (2 * sigma**2))
        
        return weight
    
    def _create_cosine_blend(self, w, h):
        """Blending cosseno - transição suave"""
        # Criar coordenadas normalizadas (0 a 1)
        y_coords, x_coords = np.meshgrid(
            np.linspace(0, 1, h),
            np.linspace(0, 1, w),
            indexing='ij'
        )
        
        # Calcular distância do centro
        center_y, center_x = 0.5, 0.5
        distance = np.sqrt((y_coords - center_y)**2 + (x_coords - center_x)**2)
        
        # Normalizar
        max_distance = np.sqrt(0.5**2 + 0.5**2)
        distance = distance / max_distance
        
        # Aplicar função cosseno
        weight = np.cos(distance * np.pi / 2)
        weight = np.clip(weight, 0, 1)
        
        return weight
    
    def _create_linear_blend(self, w, h):
        """Blending linear - transição mais abrupta"""
        # Criar coordenadas normalizadas
        y_coords, x_coords = np.meshgrid(
            np.linspace(0, 1, h),
            np.linspace(0, 1, w),
            indexing='ij'
        )
        
        # Calcular distância do centro
        center_y, center_x = 0.5, 0.5
        distance = np.sqrt((y_coords - center_y)**2 + (x_coords - center_x)**2)
        
        # Normalizar
        max_distance = np.sqrt(0.5**2 + 0.5**2)
        distance = distance / max_distance
        
        # Aplicar função linear
        weight = 1 - distance
        weight = np.clip(weight, 0, 1)
        
        return weight
    
    def _create_exponential_blend(self, w, h):
        """Blending exponencial - muito peso no centro com correção de coordenadas"""
        # Criar coordenadas normalizadas (-1 a 1 para simetria)
        y_coords, x_coords = np.meshgrid(
            np.linspace(-1, 1, h),
            np.linspace(-1, 1, w),
            indexing='ij'
        )
        
        # Calcular distância do centro (0,0)
        distance = np.sqrt(x_coords**2 + y_coords**2)
        
        # Normalizar para 0-1
        max_distance = np.sqrt(2)  # Distância máxima de (-1,-1) a (1,1)
        distance = distance / max_distance
        
        # Aplicar função exponencial com controle de suavidade
        weight = np.exp(-2.5 * distance)  # Ajustado para melhor distribuição
        weight = np.clip(weight, 0, 1)
        
        # Aplicar suavização adicional nas bordas para evitar artefatos
        feather_size = min(w, h) // 12
        for i in range(feather_size):
            # Borda superior
            edge_weight = i / feather_size
            weight[i, :] = weight[i, :] * edge_weight
            # Borda inferior
            weight[h-1-i, :] = weight[h-1-i, :] * edge_weight
            # Borda esquerda
            weight[:, i] = weight[:, i] * edge_weight
            # Borda direita
            weight[:, w-1-i] = weight[:, w-1-i] * edge_weight
        
        return weight
    
    def _create_radial_blend(self, w, h):
        """Blending radial - simétrico em todas as direções"""
        # Criar coordenadas polares
        center_y, center_x = h // 2, w // 2
        
        y_coords, x_coords = np.meshgrid(
            np.arange(h),
            np.arange(w),
            indexing='ij'
        )
        
        # Calcular distância radial do centro
        distance = np.sqrt((y_coords - center_y)**2 + (x_coords - center_x)**2)
        
        # Normalizar para 0-1
        max_distance = np.sqrt(center_y**2 + center_x**2)
        distance = distance / max_distance
        
        # Aplicar função radial com controle de suavidade
        weight = np.exp(-2.0 * distance)
        weight = np.clip(weight, 0, 1)
        
        # Suavização adicional nas bordas
        feather_size = min(w, h) // 10
        for i in range(feather_size):
            # Todas as bordas
            edge_weight = i / feather_size
            weight[i, :] = weight[i, :] * edge_weight
            weight[h-1-i, :] = weight[h-1-i, :] * edge_weight
            weight[:, i] = weight[:, i] * edge_weight
            weight[:, w-1-i] = weight[:, w-1-i] * edge_weight
        
        return weight
    
    def _generate_windows(self, h, w):
        """Gera coordenadas das janelas"""
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dir', type=str, required=True, help='path to test image or directory')
    parser.add_argument('--out_dir', type=str, required=True, help='path to output directory')
    parser.add_argument('--pretrained', type=str, required=True, help='path to pretrained model')
    parser.add_argument('--window_size', type=int, default=1024, help='window size for processing')
    parser.add_argument('--stride', type=int, default=512, help='stride between windows')
    parser.add_argument('--blend_type', type=str, default='gaussian', 
                       choices=['gaussian', 'cosine', 'linear', 'exponential', 'radial'],
                       help='type of blending function')
    args = parser.parse_args()

    # Configurar GPU
    if torch.cuda.is_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        torch.cuda.set_device(0)
        print(f"Usando GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Usando CPU")
    
    # Configurar seed
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    print(f"Usando blending: {args.blend_type}")
    
    # Criar preditor
    predictor = PanoramicPredictorAdvanced(
        model_path=args.pretrained,
        window_size=(args.window_size, args.window_size),
        stride=(args.stride, args.stride),
        blend_type=args.blend_type
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
    main()
