#!/usr/bin/env python3
"""
Predição com correção específica para vazamento de pixels
"""

import numpy as np
import argparse
from tqdm import tqdm
import cv2
import os
import torch
import gc

from SpA_Former import Generator


class PanoramicPredictorFixed:
    """Preditor com correção de vazamento de pixels"""
    
    def __init__(self, model_path, window_size=(1024, 1024), stride=(512, 512)):
        self.window_size = window_size
        self.stride = stride
        
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
        """Processa uma imagem panorâmica com correção de vazamento"""
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
                
                # Extrair janela com verificação rigorosa
                window, valid_coords = self._extract_window_safe(img_normalized, x, y, h, w)
                if window is None:
                    continue
                
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
                    
                    # Aplicar blending inteligente
                    blend_window = self._create_smart_blend(window.shape[1], window.shape[0])
                    
                    # Aplicar resultado apenas na região válida
                    self._apply_window_safe(output_img, count_map, output_window, blend_window, valid_coords)
                    
                except Exception as e:
                    print(f"❌ Erro na janela {i}: {e}")
                    # Usar imagem original para esta janela
                    self._apply_window_safe(output_img, count_map, window, np.ones_like(blend_window), valid_coords)
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
    
    def _extract_window_safe(self, img, x, y, h, w):
        """Extrai janela com verificação rigorosa de limites"""
        # Calcular limites da janela
        end_x = min(x + self.window_size[0], w)
        end_y = min(y + self.window_size[1], h)
        
        # Verificar se a janela é válida
        if end_x <= x or end_y <= y:
            return None, None
        
        # Extrair janela
        window = img[y:end_y, x:end_x]
        
        # Se a janela for menor que o esperado, redimensionar
        if window.shape[0] != self.window_size[1] or window.shape[1] != self.window_size[0]:
            window = cv2.resize(window, (self.window_size[0], self.window_size[1]))
        
        # Retornar janela e coordenadas válidas
        valid_coords = (x, y, end_x, end_y)
        return window, valid_coords
    
    def _apply_window_safe(self, output_img, count_map, window, blend, valid_coords):
        """Aplica resultado apenas na região válida"""
        x, y, end_x, end_y = valid_coords
        
        # Redimensionar blending para o tamanho real da região
        actual_h, actual_w = end_y - y, end_x - x
        if blend.shape[0] != actual_h or blend.shape[1] != actual_w:
            blend = cv2.resize(blend, (actual_w, actual_h))
            window = cv2.resize(window, (actual_w, actual_h))
        
        # Aplicar apenas na região válida
        output_img[y:end_y, x:end_x] += window * blend[:, :, np.newaxis]
        count_map[y:end_y, x:end_x] += blend
    
    def _create_smart_blend(self, w, h):
        """Cria blending inteligente que evita vazamento"""
        # Criar blending baseado em distância do centro
        center_y, center_x = h // 2, w // 2
        
        y_coords, x_coords = np.meshgrid(
            np.arange(h),
            np.arange(w),
            indexing='ij'
        )
        
        # Calcular distância do centro
        distance = np.sqrt((y_coords - center_y)**2 + (x_coords - center_x)**2)
        
        # Normalizar
        max_distance = np.sqrt(center_y**2 + center_x**2)
        distance = distance / max_distance
        
        # Aplicar função de peso com controle de vazamento
        weight = np.exp(-2.5 * distance)
        weight = np.clip(weight, 0, 1)
        
        # Aplicar suavização mais agressiva nas bordas para evitar vazamento
        feather_size = min(w, h) // 6  # Suavização mais forte
        
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
    
    def _generate_windows(self, h, w):
        """Gera coordenadas das janelas com verificação de limites"""
        windows = []
        
        # Janelas principais
        for y in range(0, h - self.window_size[1] + 1, self.stride[1]):
            for x in range(0, w - self.window_size[0] + 1, self.stride[0]):
                # Verificar se a janela está completamente dentro da imagem
                if x + self.window_size[0] <= w and y + self.window_size[1] <= h:
                    windows.append((x, y))
        
        # Janelas finais para cobrir bordas (apenas se necessário)
        if h > self.window_size[1]:
            y = h - self.window_size[1]
            for x in range(0, w - self.window_size[0] + 1, self.stride[0]):
                if x + self.window_size[0] <= w:
                    windows.append((x, y))
        
        if w > self.window_size[0]:
            x = w - self.window_size[0]
            for y in range(0, h - self.window_size[1] + 1, self.stride[1]):
                if y + self.window_size[1] <= h:
                    windows.append((x, y))
        
        # Janela do canto inferior direito
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
    
    print("Usando blending inteligente com correção de vazamento")
    
    # Criar preditor
    predictor = PanoramicPredictorFixed(
        model_path=args.pretrained,
        window_size=(args.window_size, args.window_size),
        stride=(args.stride, args.stride)
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
