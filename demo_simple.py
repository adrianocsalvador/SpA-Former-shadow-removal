#!/usr/bin/env python3
"""
Demonstração simplificada do sistema de janela deslizante
"""

import os
import cv2
import numpy as np
import yaml

class SimpleConfig:
    """Configuração simples sem attrdict"""
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)

def load_config(config_file):
    """Carrega configuração do arquivo YAML"""
    if not os.path.exists(config_file):
        print(f"Arquivo de configuração {config_file} não encontrado!")
        return None
    
    with open(config_file, 'r', encoding='UTF-8') as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)
    
    return SimpleConfig(config_dict)

def test_sliding_window_dataset(config, window_size=(640, 480), stride=(320, 240)):
    """Testa o dataset de janela deslizante"""
    
    print("=== Testando Dataset de Janela Deslizante ===")
    
    # Verificar se o dataset existe
    if not os.path.exists(config.datasets_dir):
        print(f"Dataset não encontrado em {config.datasets_dir}")
        return False
    
    train_a_dir = os.path.join(config.datasets_dir, 'train_A')
    train_c_dir = os.path.join(config.datasets_dir, 'train_C')
    
    if not os.path.exists(train_a_dir) or not os.path.exists(train_c_dir):
        print("Pastas train_A ou train_C não encontradas!")
        return False
    
    # Listar imagens
    images = [f for f in os.listdir(train_a_dir) if f.endswith('.png')]
    print(f"Encontradas {len(images)} imagens no dataset")
    
    if not images:
        print("Nenhuma imagem encontrada!")
        return False
    
    # Testar algumas imagens
    total_windows = 0
    
    for i, img_name in enumerate(images[:3]):  # Testar apenas 3 imagens
        print(f"\n--- Testando imagem {i+1}: {img_name} ---")
        
        # Carregar imagem
        img_path = os.path.join(train_a_dir, img_name)
        target_path = os.path.join(train_c_dir, img_name)
        
        img = cv2.imread(img_path, 1)
        target = cv2.imread(target_path, 1)
        
        if img is None or target is None:
            print(f"Erro ao carregar {img_name}")
            continue
        
        h, w = img.shape[:2]
        print(f"Tamanho da imagem: {w}x{h}")
        
        # Gerar janelas
        windows = []
        for y in range(0, h - window_size[1] + 1, stride[1]):
            for x in range(0, w - window_size[0] + 1, stride[0]):
                windows.append((x, y))
        
        # Adicionar janelas finais
        if h > window_size[1]:
            y = h - window_size[1]
            for x in range(0, w - window_size[0] + 1, stride[0]):
                windows.append((x, y))
        
        if w > window_size[0]:
            x = w - window_size[0]
            for y in range(0, h - window_size[1] + 1, stride[1]):
                windows.append((x, y))
        
        if h > window_size[1] and w > window_size[0]:
            windows.append((w - window_size[0], h - window_size[1]))
        
        # Remover duplicatas
        windows = list(set(windows))
        
        print(f"Janelas geradas: {len(windows)}")
        total_windows += len(windows)
        
        # Testar extração de uma janela
        if windows:
            x, y = windows[0]
            img_window = img[y:y+window_size[1], x:x+window_size[0]]
            target_window = target[y:y+window_size[1], x:x+window_size[0]]
            
            # Calcular máscara
            M = np.clip((target_window.astype(np.float32) - img_window.astype(np.float32)).sum(axis=2), 0, 1)
            
            print(f"Janela extraída: posição ({x},{y})")
            print(f"  - Input shape: {img_window.shape}")
            print(f"  - Target shape: {target_window.shape}")
            print(f"  - Mask range: [{M.min():.3f}, {M.max():.3f}]")
    
    print(f"\nTotal de janelas em todas as imagens: {total_windows}")
    return True

def main():
    """Função principal"""
    
    print("🚀 Demonstração do Sistema de Janela Deslizante")
    print("=" * 50)
    
    # Carregar configuração
    config_files = ['config_sliding_window.yml', 'config.yml']
    config = None
    
    for config_file in config_files:
        if os.path.exists(config_file):
            config = load_config(config_file)
            print(f"✅ Configuração carregada: {config_file}")
            break
    
    if config is None:
        print("❌ Nenhuma configuração encontrada!")
        return
    
    # Configurar parâmetros
    window_size = (getattr(config, 'window_width', 640), getattr(config, 'window_height', 480))
    stride = (getattr(config, 'stride_width', 320), getattr(config, 'stride_height', 240))
    
    print(f"Tamanho da janela: {window_size[0]}x{window_size[1]}")
    print(f"Stride: {stride[0]}x{stride[1]}")
    
    # Testar dataset
    success = test_sliding_window_dataset(config, window_size, stride)
    
    if success:
        print("\n🎉 Demonstração concluída com sucesso!")
        print("O sistema de janela deslizante está funcionando corretamente.")
        print("\nPróximo passo: iniciar treinamento")
        print("python3 start_sliding_window_training.py")
    else:
        print("\n❌ Demonstração falhou!")
        print("Verifique a configuração do dataset.")

if __name__ == '__main__':
    main()
