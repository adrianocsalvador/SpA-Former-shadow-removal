#!/usr/bin/env python3
"""
Teste com janelas pequenas para otimizar velocidade e qualidade
"""

import numpy as np
import argparse
from tqdm import tqdm
import cv2
import os
import torch
import gc
import time

from SpA_Former import Generator


class SmallWindowPredictor:
    """Preditor otimizado para janelas pequenas"""
    
    def __init__(self, model_path, window_size=(256, 256), stride=(128, 128)):
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
        """Processa uma imagem com janelas pequenas"""
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
        
        # Medir tempo
        start_time = time.time()
        
        # Processar cada janela
        with torch.no_grad():
            for i, (x, y) in enumerate(tqdm(windows, desc="Processando janelas")):
                # Limpar cache a cada 50 janelas
                if i % 50 == 0:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                
                # Extrair janela
                end_x = min(x + self.window_size[0], w)
                end_y = min(y + self.window_size[1], h)
                
                if end_x <= x or end_y <= y:
                    continue
                
                window = img_normalized[y:end_y, x:end_x]
                
                # Redimensionar se necessário
                if window.shape[0] != self.window_size[1] or window.shape[1] != self.window_size[0]:
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
                        output_window = window
                    
                    output_window = np.clip(output_window, 0, 1)
                    
                    # Aplicar blending simples
                    blend_window = self._create_simple_blend(self.window_size[0], self.window_size[1])
                    
                    # Redimensionar para o tamanho real
                    actual_h, actual_w = end_y - y, end_x - x
                    if blend_window.shape[0] != actual_h or blend_window.shape[1] != actual_w:
                        blend_window = cv2.resize(blend_window, (actual_w, actual_h))
                        output_window = cv2.resize(output_window, (actual_w, actual_h))
                    
                    # Aplicar à imagem de saída
                    output_img[y:end_y, x:end_x] += output_window * blend_window[:, :, np.newaxis]
                    count_map[y:end_y, x:end_x] += blend_window
                    
                except Exception as e:
                    print(f"❌ Erro na janela {i}: {e}")
                    continue
                
                # Liberar memória
                del window_tensor, output_window
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Calcular tempo total
        total_time = time.time() - start_time
        
        # Normalizar pela contagem
        count_map[count_map == 0] = 1
        output_img = output_img / count_map[:, :, np.newaxis]
        
        # Verificar valores finais
        if np.any(np.isnan(output_img)) or np.any(np.isinf(output_img)):
            output_img = img_normalized
        
        # Converter de volta para uint8
        output_img = np.clip(output_img * 255, 0, 255).astype(np.uint8)
        
        # Salvar resultado
        cv2.imwrite(output_path, output_img)
        print(f"Resultado salvo em: {output_path}")
        print(f"Tempo total: {total_time:.2f} segundos")
        print(f"Velocidade: {len(windows)/total_time:.2f} janelas/segundo")
        
        return output_img
    
    def _create_simple_blend(self, w, h):
        """Cria blending simples para janelas pequenas"""
        blend = np.ones((h, w), dtype=np.float32)
        
        # Suavização simples nas bordas
        feather_size = min(w, h) // 8
        
        for i in range(feather_size):
            # Bordas
            edge_weight = i / feather_size
            blend[i, :] = blend[i, :] * edge_weight
            blend[h-1-i, :] = blend[h-1-i, :] * edge_weight
            blend[:, i] = blend[:, i] * edge_weight
            blend[:, w-1-i] = blend[:, w-1-i] * edge_weight
        
        return blend
    
    def _generate_windows(self, h, w):
        """Gera coordenadas das janelas"""
        windows = []
        
        # Janelas principais
        for y in range(0, h - self.window_size[1] + 1, self.stride[1]):
            for x in range(0, w - self.window_size[0] + 1, self.stride[0]):
                if x + self.window_size[0] <= w and y + self.window_size[1] <= h:
                    windows.append((x, y))
        
        # Janelas finais para cobrir bordas
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
    parser.add_argument('--window_size', type=int, default=256, help='window size for processing')
    parser.add_argument('--stride', type=int, default=128, help='stride between windows')
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
    
    print(f"Configuração: Janela {args.window_size}x{args.window_size}, Stride {args.stride}x{args.stride}")
    
    # Criar preditor
    predictor = SmallWindowPredictor(
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
