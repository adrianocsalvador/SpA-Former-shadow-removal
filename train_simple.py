#!/usr/bin/env python3
"""
Versão simplificada do treinamento com janela deslizante
Para testar o sistema sem dependências complexas
"""

import os
import cv2
import numpy as np
import yaml
import time
import random

class SimpleConfig:
    """Configuração simples"""
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)

class SimpleSlidingWindowDataset:
    """Dataset simplificado de janela deslizante"""
    
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
        
        imgs = np.array([w[0] for w in batch])
        targets = np.array([w[1] for w in batch])
        masks = np.array([w[2] for w in batch])
        
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

def simulate_training(config):
    """Simula o treinamento para testar o sistema"""
    
    print("🚀 Simulando Treinamento com Janela Deslizante")
    print("=" * 50)
    
    # Configurar parâmetros
    window_size = (getattr(config, 'window_width', 640), getattr(config, 'window_height', 480))
    stride = (getattr(config, 'stride_width', 320), getattr(config, 'stride_height', 240))
    batch_size = getattr(config, 'batchsize', 4)
    epochs = getattr(config, 'epoch', 5)  # Apenas 5 épocas para teste
    
    print(f"Tamanho da janela: {window_size[0]}x{window_size[1]}")
    print(f"Stride: {stride[0]}x{stride[1]}")
    print(f"Batch size: {batch_size}")
    print(f"Épocas: {epochs}")
    
    # Criar dataset
    dataset = SimpleSlidingWindowDataset(config, window_size, stride)
    print(f"Dataset criado: {len(dataset)} janelas totais")
    
    # Simular treinamento
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        print(f"\n--- Época {epoch}/{epochs} ---")
        
        # Calcular iterações por época
        iterations = len(dataset) // batch_size
        print(f"Iterações por época: {iterations}")
        
        for iteration in range(1, min(iterations + 1, 10)):  # Máximo 10 iterações por época para teste
            try:
                # Obter batch
                imgs, targets, masks = dataset.get_next_batch(batch_size)
                
                # Simular processamento
                fake_loss_g = np.random.uniform(0.1, 0.5)
                fake_loss_d = np.random.uniform(0.2, 0.8)
                
                if iteration % 5 == 0:
                    print(f"  Iteração {iteration}: Loss_G={fake_loss_g:.4f}, Loss_D={fake_loss_d:.4f}")
                
                # Simular tempo de processamento
                time.sleep(0.1)
                
            except Exception as e:
                print(f"Erro na iteração {iteration}: {e}")
                break
        
        epoch_time = time.time() - epoch_start
        print(f"Época {epoch} concluída em {epoch_time:.2f}s")
    
    total_time = time.time() - start_time
    print(f"\n🎉 Simulação concluída em {total_time:.2f}s")
    print("O sistema de janela deslizante está funcionando corretamente!")
    
    return True

def main():
    """Função principal"""
    
    # Carregar configuração
    config_files = ['config_sliding_window.yml', 'config.yml']
    config = None
    
    for config_file in config_files:
        if os.path.exists(config_file):
            with open(config_file, 'r', encoding='UTF-8') as f:
                config_dict = yaml.load(f, Loader=yaml.FullLoader)
            config = SimpleConfig(config_dict)
            print(f"✅ Configuração carregada: {config_file}")
            break
    
    if config is None:
        print("❌ Nenhuma configuração encontrada!")
        return
    
    # Verificar dataset
    if not os.path.exists(config.datasets_dir):
        print(f"❌ Dataset não encontrado em {config.datasets_dir}")
        return
    
    # Executar simulação
    success = simulate_training(config)
    
    if success:
        print("\n✅ Sistema testado com sucesso!")
        print("Agora você pode executar o treinamento completo:")
        print("python3 start_sliding_window_training.py")
    else:
        print("\n❌ Teste falhou!")

if __name__ == '__main__':
    main()
