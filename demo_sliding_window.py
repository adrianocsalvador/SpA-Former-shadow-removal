#!/usr/bin/env python3
"""
Demonstração do sistema de janela deslizante para SpA-Former
Mostra como as janelas são criadas em tempo real a partir de imagens originais
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from sliding_window_train import SlidingWindowDataset
import yaml
from attrdict import AttrMap

def visualize_sliding_windows(config, image_name, window_size=(640, 480), stride=(320, 240)):
    """
    Visualiza as janelas deslizantes criadas a partir de uma imagem
    """
    # Carregar imagem original
    img_path = os.path.join(config.datasets_dir, 'train_A', image_name)
    target_path = os.path.join(config.datasets_dir, 'train_C', image_name)
    
    if not os.path.exists(img_path) or not os.path.exists(target_path):
        print(f"Erro: Imagens não encontradas para {image_name}")
        return
    
    # Carregar imagens
    img = cv2.imread(img_path, 1)
    target = cv2.imread(target_path, 1)
    
    if img is None or target is None:
        print(f"Erro: Não foi possível carregar as imagens para {image_name}")
        return
    
    # Converter BGR para RGB para matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    target_rgb = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
    
    h, w = img.shape[:2]
    print(f"Tamanho da imagem original: {w}x{h}")
    print(f"Tamanho da janela: {window_size[0]}x{window_size[1]}")
    print(f"Stride: {stride[0]}x{stride[1]}")
    
    # Gerar coordenadas das janelas
    windows = []
    for y in range(0, h - window_size[1] + 1, stride[1]):
        for x in range(0, w - window_size[0] + 1, stride[0]):
            windows.append((x, y))
    
    # Adicionar janelas finais se necessário
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
    
    print(f"Total de janelas geradas: {len(windows)}")
    
    # Criar visualização
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Demonstração: Janelas Deslizantes - {image_name}', fontsize=16)
    
    # Mostrar imagem original
    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title('Imagem Original (com sombra)')
    axes[0, 0].axis('off')
    
    # Mostrar target
    axes[0, 1].imshow(target_rgb)
    axes[0, 1].set_title('Imagem Target (sem sombra)')
    axes[0, 1].axis('off')
    
    # Mostrar diferença (máscara de sombra)
    diff = np.clip((target.astype(np.float32) - img.astype(np.float32)).sum(axis=2), 0, 255).astype(np.uint8)
    axes[0, 2].imshow(diff, cmap='gray')
    axes[0, 2].set_title('Máscara de Sombra')
    axes[0, 2].axis('off')
    
    # Mostrar algumas janelas
    for i, (x, y) in enumerate(windows[:3]):
        row = 1
        col = i
        
        # Extrair janela
        window_img = img_rgb[y:y+window_size[1], x:x+window_size[0]]
        window_target = target_rgb[y:y+window_size[1], x:x+window_size[0]]
        
        # Desenhar retângulo na imagem original
        img_with_rect = img_rgb.copy()
        cv2.rectangle(img_with_rect, (x, y), (x+window_size[0], y+window_size[1]), (255, 0, 0), 3)
        
        axes[row, col].imshow(window_img)
        axes[row, col].set_title(f'Janela {i+1}: ({x},{y})')
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Salvar visualização
    output_path = f'sliding_window_demo_{image_name.replace(".png", "")}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualização salva em: {output_path}")
    
    return windows

def test_sliding_window_dataset(config, window_size=(640, 480), stride=(320, 240)):
    """
    Testa o dataset de janela deslizante
    """
    print("=== Testando Dataset de Janela Deslizante ===")
    
    # Criar dataset
    dataset = SlidingWindowDataset(config, window_size, stride)
    
    print(f"Total de janelas no dataset: {len(dataset)}")
    
    # Testar alguns batches
    for i in range(3):
        print(f"\n--- Batch {i+1} ---")
        try:
            imgs, targets, masks = dataset.get_next_batch(2)
            print(f"Shape das imagens: {imgs.shape}")
            print(f"Shape dos targets: {targets.shape}")
            print(f"Shape das máscaras: {masks.shape}")
            print(f"Range das imagens: [{imgs.min():.3f}, {imgs.max():.3f}]")
            print(f"Range dos targets: [{targets.min():.3f}, {targets.max():.3f}]")
            print(f"Range das máscaras: [{masks.min():.3f}, {masks.max():.3f}]")
        except Exception as e:
            print(f"Erro ao obter batch: {e}")
            break

def main():
    parser = argparse.ArgumentParser(description='Demonstração do Sistema de Janela Deslizante')
    parser.add_argument('--config', type=str, default='config_sliding_window.yml',
                       help='Arquivo de configuração')
    parser.add_argument('--image', type=str, default='',
                       help='Nome da imagem para visualizar (ex: 100-6.png)')
    parser.add_argument('--window_size', type=int, nargs=2, default=[640, 480],
                       help='Tamanho da janela (width height)')
    parser.add_argument('--stride', type=int, nargs=2, default=[320, 240],
                       help='Stride da janela (width height)')
    parser.add_argument('--test_dataset', action='store_true',
                       help='Testar o dataset de janela deslizante')
    
    args = parser.parse_args()
    
    # Carregar configuração
    if not os.path.exists(args.config):
        print(f"Erro: Arquivo de configuração {args.config} não encontrado!")
        return
    
    with open(args.config, 'r', encoding='UTF-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    config = AttrMap(config)
    
    # Verificar se o dataset existe
    if not os.path.exists(config.datasets_dir):
        print(f"Erro: Dataset não encontrado em {config.datasets_dir}")
        return
    
    train_a_dir = os.path.join(config.datasets_dir, 'train_A')
    if not os.path.exists(train_a_dir):
        print(f"Erro: Pasta train_A não encontrada em {train_a_dir}")
        return
    
    # Listar imagens disponíveis
    available_images = [f for f in os.listdir(train_a_dir) if f.endswith('.png')]
    print(f"Imagens disponíveis: {len(available_images)}")
    
    if args.test_dataset:
        test_sliding_window_dataset(config, args.window_size, args.stride)
    
    if args.image:
        if args.image not in available_images:
            print(f"Erro: Imagem {args.image} não encontrada!")
            print("Imagens disponíveis:")
            for img in available_images[:10]:  # Mostrar apenas as primeiras 10
                print(f"  - {img}")
            return
        
        visualize_sliding_windows(config, args.image, args.window_size, args.stride)
    else:
        # Usar primeira imagem disponível
        if available_images:
            first_image = available_images[0]
            print(f"Usando primeira imagem disponível: {first_image}")
            visualize_sliding_windows(config, first_image, args.window_size, args.stride)
        else:
            print("Nenhuma imagem encontrada no dataset!")

if __name__ == '__main__':
    main()
