#!/usr/bin/env python3
"""
Teste do modelo treinado com dataset_v9 em imagens panorâmicas
"""

import os
import cv2
import numpy as np
import torch
import argparse
import time
from pathlib import Path
from SpA_Former import Generator


class TrainedModelPredictor:
    """Predictor usando modelo treinado com dataset_v9"""
    
    def __init__(self, model_path, window_size=(512, 512), stride=(256, 256)):
        self.window_size = window_size
        self.stride = stride
        
        # Carregar modelo treinado
        print(f"🔄 Carregando modelo treinado: {model_path}")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Inicializar modelo
        self.model = Generator(gpu_ids=[0] if torch.cuda.is_available() else [])
        
        # Carregar pesos treinados
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model.eval()
        
        if torch.cuda.is_available():
            self.model.cuda()
            print("✅ Modelo carregado na GPU")
        else:
            print("⚠️ Modelo carregado na CPU")
    
    def process_image(self, image_path, output_path):
        """Processa imagem panorâmica com modelo treinado"""
        print(f"🔄 Processando: {image_path}")
        
        # Carregar imagem
        img = cv2.imread(image_path)
        if img is None:
            print(f"❌ Erro ao carregar imagem: {image_path}")
            return
        
        h, w = img.shape[:2]
        print(f"📏 Tamanho da imagem: {w}x{h}")
        
        # Inicializar imagem de saída
        output_img = np.zeros_like(img, dtype=np.float32)
        count_map = np.zeros((h, w), dtype=np.float32)
        
        # Gerar coordenadas das janelas
        windows = self._generate_windows(h, w)
        print(f"🪟 Total de janelas: {len(windows)}")
        
        start_time = time.time()
        
        for i, (x, y) in enumerate(windows):
            if i % 10 == 0:
                print(f"🔄 Processando janela {i+1}/{len(windows)}")
            
            # Extrair janela
            end_y = min(y + self.window_size[1], h)
            end_x = min(x + self.window_size[0], w)
            
            window = img[y:end_y, x:end_x]
            
            # Redimensionar se necessário
            if window.shape[:2] != self.window_size:
                window = cv2.resize(window, self.window_size)
            
            # Normalizar
            window_normalized = window.astype(np.float32) / 255.0
            
            # Converter para tensor
            window_tensor = torch.from_numpy(window_normalized.transpose(2, 0, 1)).unsqueeze(0)
            
            if torch.cuda.is_available():
                window_tensor = window_tensor.cuda()
            
            # Predição
            with torch.no_grad():
                try:
                    att, output_window = self.model(window_tensor)
                    output_window = output_window.squeeze(0).cpu().numpy().transpose(1, 2, 0)
                except Exception as e:
                    print(f"⚠️ Erro na predição da janela {i}: {e}")
                    output_window = window_normalized
            
            # Verificar valores inválidos
            if np.any(np.isnan(output_window)) or np.any(np.isinf(output_window)):
                print(f"⚠️ Valores inválidos na janela {i}, usando imagem original")
                output_window = window_normalized
            
            # Clipping
            output_window = np.clip(output_window, 0, 1)
            
            # Criar janela de blending
            blend_window = self._create_smart_blend(window.shape[1], window.shape[0])
            
            # Aplicar resultado
            actual_h, actual_w = window.shape[:2]
            output_img[y:y+actual_h, x:x+actual_w] += output_window[:actual_h, :actual_w] * blend_window[:actual_h, :actual_w, np.newaxis]
            count_map[y:y+actual_h, x:x+actual_w] += blend_window[:actual_h, :actual_w]
            
            # Limpar cache a cada 10 janelas
            if i % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Normalizar resultado final
        count_map[count_map == 0] = 1  # Evitar divisão por zero
        output_img = output_img / count_map[:, :, np.newaxis]
        
        # Converter para uint8
        output_img = np.clip(output_img * 255, 0, 255).astype(np.uint8)
        
        # Salvar resultado
        cv2.imwrite(output_path, output_img)
        
        total_time = time.time() - start_time
        print(f"✅ Processamento concluído em {total_time:.2f}s")
        print(f"📁 Resultado salvo em: {output_path}")
        
        return output_img
    
    def _generate_windows(self, h, w):
        """Gera coordenadas das janelas"""
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
        
        return list(set(windows))
    
    def _create_smart_blend(self, w, h):
        """Cria janela de blending inteligente"""
        # Criar coordenadas normalizadas
        y_coords = np.linspace(-1, 1, h)
        x_coords = np.linspace(-1, 1, w)
        Y, X = np.meshgrid(y_coords, x_coords, indexing='ij')
        
        # Distância do centro
        distance = np.sqrt(X**2 + Y**2)
        
        # Blending com feathering mais forte
        feather_size = min(w, h) // 8
        blend = np.exp(-distance * feather_size)
        
        # Normalizar
        blend = (blend - blend.min()) / (blend.max() - blend.min())
        
        return blend


def main():
    parser = argparse.ArgumentParser(description='Teste do modelo treinado com dataset_v9')
    parser.add_argument('--model_path', type=str, required=True, 
                       help='Caminho para o modelo treinado')
    parser.add_argument('--test_image', type=str, required=True,
                       help='Caminho para imagem de teste')
    parser.add_argument('--output_dir', type=str, default='./test_trained_output',
                       help='Diretório de saída')
    parser.add_argument('--window_size', type=int, default=512,
                       help='Tamanho da janela')
    parser.add_argument('--stride', type=int, default=256,
                       help='Stride da janela')
    
    args = parser.parse_args()
    
    # Criar diretório de saída
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Inicializar predictor
    predictor = TrainedModelPredictor(
        model_path=args.model_path,
        window_size=(args.window_size, args.window_size),
        stride=(args.stride, args.stride)
    )
    
    # Processar imagem
    output_path = os.path.join(args.output_dir, f"trained_result_{Path(args.test_image).stem}.png")
    predictor.process_image(args.test_image, output_path)
    
    print(f"🎉 Teste concluído! Resultado salvo em: {output_path}")


if __name__ == "__main__":
    main()
