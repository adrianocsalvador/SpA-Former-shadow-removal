#!/usr/bin/env python3
"""
Preparação do dataset_v9_small com apenas 5 imagens de cada tipo
Organiza imagens com sombra (.jpeg) e sem sombra (.jpg)
"""

import os
import cv2
import numpy as np
import shutil
from pathlib import Path
import argparse
from tqdm import tqdm


def create_sliding_windows(image, window_size=(512, 512), stride=(256, 256)):
    """Cria janelas deslizantes da imagem"""
    h, w = image.shape[:2]
    windows = []
    coordinates = []
    
    for y in range(0, h - window_size[1] + 1, stride[1]):
        for x in range(0, w - window_size[0] + 1, stride[0]):
            window = image[y:y+window_size[1], x:x+window_size[0]]
            windows.append(window)
            coordinates.append((x, y))
    
    # Adicionar janelas finais para cobrir bordas
    if h > window_size[1]:
        y = h - window_size[1]
        for x in range(0, w - window_size[0] + 1, stride[0]):
            window = image[y:y+window_size[1], x:x+window_size[0]]
            windows.append(window)
            coordinates.append((x, y))
    
    if w > window_size[0]:
        x = w - window_size[0]
        for y in range(0, h - window_size[1] + 1, stride[1]):
            window = image[y:y+window_size[1], x:x+window_size[0]]
            windows.append(window)
            coordinates.append((x, y))
    
    if h > window_size[1] and w > window_size[0]:
        window = image[h-window_size[1]:h, w-window_size[0]:w]
        windows.append(window)
        coordinates.append((w-window_size[0], h-window_size[1]))
    
    return windows, coordinates


def prepare_dataset_v9_small():
    """Prepara o dataset_v9_small com apenas 5 imagens de cada tipo"""
    
    # Configurações
    base_dir = "/mnt/48EC7EE9EC7ED0A4/Teste_Sombra"
    dataset_dir = os.path.join(base_dir, "dataset_v9_small")
    shadow_dir = os.path.join(base_dir, "20250717_01/PAN")  # Imagens com sombra (.jpeg)
    no_shadow_dir = os.path.join(base_dir, "20250717_01/PAN_SOMBRA")  # Imagens sem sombra (.jpg)
    
    window_size = (512, 512)
    stride = (256, 256)  # 50% overlap
    
    # Criar estrutura de diretórios
    train_a_dir = os.path.join(dataset_dir, "train", "train_A")
    train_c_dir = os.path.join(dataset_dir, "train", "train_C")
    test_a_dir = os.path.join(dataset_dir, "test", "test_A")
    test_c_dir = os.path.join(dataset_dir, "test", "test_C")
    
    for dir_path in [train_a_dir, train_c_dir, test_a_dir, test_c_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    print(f"📁 Criando dataset pequeno em: {dataset_dir}")
    print(f"📂 Imagens com sombra: {shadow_dir}")
    print(f"📂 Imagens sem sombra: {no_shadow_dir}")
    print(f"🪟 Tamanho da janela: {window_size}")
    print(f"📏 Stride: {stride}")
    
    # Listar arquivos
    shadow_files = [f for f in os.listdir(shadow_dir) if f.endswith('.jpeg')]
    no_shadow_files = [f for f in os.listdir(no_shadow_dir) if f.endswith('.jpg')]
    
    print(f"📊 Encontrados {len(shadow_files)} arquivos com sombra (.jpeg)")
    print(f"📊 Encontrados {len(no_shadow_files)} arquivos sem sombra (.jpg)")
    
    # Pegar apenas 5 imagens de cada tipo
    shadow_files = shadow_files[:5]
    no_shadow_files = no_shadow_files[:5]
    
    print(f"🎯 Usando apenas 5 imagens de cada tipo para teste inicial")
    print(f"📋 Imagens com sombra: {shadow_files}")
    print(f"📋 Imagens sem sombra: {no_shadow_files}")
    
    # Separar para treino e teste (80% treino, 20% teste)
    # Com 5 imagens: 4 para treino, 1 para teste
    train_shadow = shadow_files[:4]
    test_shadow = shadow_files[4:]
    train_no_shadow = no_shadow_files[:4]
    test_no_shadow = no_shadow_files[4:]
    
    print(f"🎯 Treino: {len(train_shadow)} com sombra, {len(train_no_shadow)} sem sombra")
    print(f"🧪 Teste: {len(test_shadow)} com sombra, {len(test_no_shadow)} sem sombra")
    
    total_windows = 0
    
    # Processar imagens de treino
    print("\n🔄 Processando imagens de TREINO...")
    
    # Processar imagens com sombra (train_A)
    for filename in tqdm(train_shadow, desc="Processando imagens com sombra (treino)"):
        img_path = os.path.join(shadow_dir, filename)
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"⚠️ Não foi possível carregar: {img_path}")
            continue
        
        print(f"📏 Processando {filename}: {img.shape[1]}x{img.shape[0]}")
        
        windows, coordinates = create_sliding_windows(img, window_size, stride)
        
        for i, (window, (x, y)) in enumerate(zip(windows, coordinates)):
            # Nome único para a janela
            base_name = Path(filename).stem
            window_name = f"{base_name}_win_{i:04d}_x{x}_y{y}.png"
            window_path = os.path.join(train_a_dir, window_name)
            
            cv2.imwrite(window_path, window)
            total_windows += 1
        
        print(f"   ✅ {filename}: {len(windows)} janelas criadas")
    
    # Processar imagens sem sombra (train_C)
    for filename in tqdm(train_no_shadow, desc="Processando imagens sem sombra (treino)"):
        img_path = os.path.join(no_shadow_dir, filename)
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"⚠️ Não foi possível carregar: {img_path}")
            continue
        
        print(f"📏 Processando {filename}: {img.shape[1]}x{img.shape[0]}")
        
        windows, coordinates = create_sliding_windows(img, window_size, stride)
        
        for i, (window, (x, y)) in enumerate(zip(windows, coordinates)):
            # Nome único para a janela
            base_name = Path(filename).stem
            window_name = f"{base_name}_win_{i:04d}_x{x}_y{y}.png"
            window_path = os.path.join(train_c_dir, window_name)
            
            cv2.imwrite(window_path, window)
            total_windows += 1
        
        print(f"   ✅ {filename}: {len(windows)} janelas criadas")
    
    # Processar imagens de teste
    print("\n🔄 Processando imagens de TESTE...")
    
    # Processar imagens com sombra (test_A)
    for filename in tqdm(test_shadow, desc="Processando imagens com sombra (teste)"):
        img_path = os.path.join(shadow_dir, filename)
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"⚠️ Não foi possível carregar: {img_path}")
            continue
        
        print(f"📏 Processando {filename}: {img.shape[1]}x{img.shape[0]}")
        
        windows, coordinates = create_sliding_windows(img, window_size, stride)
        
        for i, (window, (x, y)) in enumerate(zip(windows, coordinates)):
            # Nome único para a janela
            base_name = Path(filename).stem
            window_name = f"{base_name}_win_{i:04d}_x{x}_y{y}.png"
            window_path = os.path.join(test_a_dir, window_name)
            
            cv2.imwrite(window_path, window)
            total_windows += 1
        
        print(f"   ✅ {filename}: {len(windows)} janelas criadas")
    
    # Processar imagens sem sombra (test_C)
    for filename in tqdm(test_no_shadow, desc="Processando imagens sem sombra (teste)"):
        img_path = os.path.join(no_shadow_dir, filename)
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"⚠️ Não foi possível carregar: {img_path}")
            continue
        
        print(f"📏 Processando {filename}: {img.shape[1]}x{img.shape[0]}")
        
        windows, coordinates = create_sliding_windows(img, window_size, stride)
        
        for i, (window, (x, y)) in enumerate(zip(windows, coordinates)):
            # Nome único para a janela
            base_name = Path(filename).stem
            window_name = f"{base_name}_win_{i:04d}_x{x}_y{y}.png"
            window_path = os.path.join(test_c_dir, window_name)
            
            cv2.imwrite(window_path, window)
            total_windows += 1
        
        print(f"   ✅ {filename}: {len(windows)} janelas criadas")
    
    # Criar arquivos de lista
    print("\n📝 Criando arquivos de lista...")
    
    # Lista de treino
    train_list_path = os.path.join(dataset_dir, "train_list.txt")
    with open(train_list_path, 'w') as f:
        for filename in os.listdir(train_a_dir):
            if filename.endswith('.png'):
                f.write(f"{filename}\n")
    
    # Lista de validação
    val_list_path = os.path.join(dataset_dir, "val_list.txt")
    with open(val_list_path, 'w') as f:
        for filename in os.listdir(test_a_dir):
            if filename.endswith('.png'):
                f.write(f"{filename}\n")
    
    # Estatísticas finais
    train_a_count = len([f for f in os.listdir(train_a_dir) if f.endswith('.png')])
    train_c_count = len([f for f in os.listdir(train_c_dir) if f.endswith('.png')])
    test_a_count = len([f for f in os.listdir(test_a_dir) if f.endswith('.png')])
    test_c_count = len([f for f in os.listdir(test_c_dir) if f.endswith('.png')])
    
    print("\n✅ Dataset pequeno criado com sucesso!")
    print(f"📊 Estatísticas finais:")
    print(f"   🎯 Treino A (com sombra): {train_a_count} janelas")
    print(f"   🎯 Treino C (sem sombra): {train_c_count} janelas")
    print(f"   🧪 Teste A (com sombra): {test_a_count} janelas")
    print(f"   🧪 Teste C (sem sombra): {test_c_count} janelas")
    print(f"   📁 Dataset salvo em: {dataset_dir}")
    print(f"   📄 Lista de treino: {train_list_path}")
    print(f"   📄 Lista de validação: {val_list_path}")
    
    return dataset_dir


if __name__ == "__main__":
    prepare_dataset_v9_small()
