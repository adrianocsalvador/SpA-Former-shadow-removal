#!/usr/bin/env python3
"""
Preparação do dataset_v9_pairs usando apenas pares correspondentes
Organiza imagens com sombra (.jpeg) e sem sombra (.jpg) que têm o mesmo número
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


def prepare_dataset_v9_pairs():
    """Prepara o dataset_v9_pairs usando apenas pares correspondentes"""
    
    # Configurações
    base_dir = "/mnt/48EC7EE9EC7ED0A4/Teste_Sombra"
    dataset_dir = os.path.join(base_dir, "dataset_v9_pairs")
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
    
    print(f"📁 Criando dataset com pares em: {dataset_dir}")
    print(f"📂 Imagens com sombra: {shadow_dir}")
    print(f"📂 Imagens sem sombra: {no_shadow_dir}")
    print(f"🪟 Tamanho da janela: {window_size}")
    print(f"📏 Stride: {stride}")
    
    # Listar arquivos
    shadow_files = [f for f in os.listdir(shadow_dir) if f.endswith('.jpeg')]
    no_shadow_files = [f for f in os.listdir(no_shadow_dir) if f.endswith('.jpg')]
    
    print(f"📊 Total de arquivos:")
    print(f"   Com sombra (.jpeg): {len(shadow_files)}")
    print(f"   Sem sombra (.jpg): {len(no_shadow_files)}")
    
    # Encontrar pares correspondentes
    shadow_numbers = {Path(f).stem: f for f in shadow_files}
    no_shadow_numbers = {Path(f).stem: f for f in no_shadow_files}
    
    matching_pairs = set(shadow_numbers.keys()).intersection(set(no_shadow_numbers.keys()))
    
    print(f"\n🔗 Pares correspondentes encontrados: {len(matching_pairs)}")
    
    if len(matching_pairs) < 5:
        print(f"❌ Poucos pares correspondentes encontrados!")
        print(f"💡 Recomendação: Verificar se os diretórios estão corretos")
        return None
    
    # Pegar apenas 5 pares para teste
    matching_pairs_list = sorted(list(matching_pairs))[:5]
    
    print(f"🎯 Usando 5 pares correspondentes para teste:")
    for i, number in enumerate(matching_pairs_list):
        print(f"   {i+1}. {number}.jpeg ↔ {number}.jpg")
    
    # Separar para treino e teste (80% treino, 20% teste)
    # Com 5 pares: 4 para treino, 1 para teste
    train_pairs = matching_pairs_list[:4]
    test_pairs = matching_pairs_list[4:]
    
    print(f"\n🎯 Treino: {len(train_pairs)} pares")
    print(f"🧪 Teste: {len(test_pairs)} pares")
    
    total_windows = 0
    
    # Processar pares de treino
    print("\n🔄 Processando pares de TREINO...")
    
    for number in tqdm(train_pairs, desc="Processando pares de treino"):
        shadow_file = shadow_numbers[number]
        no_shadow_file = no_shadow_numbers[number]
        
        shadow_path = os.path.join(shadow_dir, shadow_file)
        no_shadow_path = os.path.join(no_shadow_dir, no_shadow_file)
        
        # Carregar imagens
        shadow_img = cv2.imread(shadow_path)
        no_shadow_img = cv2.imread(no_shadow_path)
        
        if shadow_img is None or no_shadow_img is None:
            print(f"⚠️ Erro ao carregar par {number}")
            continue
        
        print(f"📏 Processando par {number}: {shadow_img.shape[1]}x{shadow_img.shape[0]}")
        
        # Criar janelas para ambas as imagens
        shadow_windows, shadow_coords = create_sliding_windows(shadow_img, window_size, stride)
        no_shadow_windows, no_shadow_coords = create_sliding_windows(no_shadow_img, window_size, stride)
        
        # Salvar janelas com sombra (train_A)
        for i, (window, (x, y)) in enumerate(zip(shadow_windows, shadow_coords)):
            window_name = f"{number}_win_{i:04d}_x{x}_y{y}.png"
            window_path = os.path.join(train_a_dir, window_name)
            cv2.imwrite(window_path, window)
        
        # Salvar janelas sem sombra (train_C)
        for i, (window, (x, y)) in enumerate(zip(no_shadow_windows, no_shadow_coords)):
            window_name = f"{number}_win_{i:04d}_x{x}_y{y}.png"
            window_path = os.path.join(train_c_dir, window_name)
            cv2.imwrite(window_path, window)
        
        total_windows += len(shadow_windows)
        print(f"   ✅ Par {number}: {len(shadow_windows)} janelas criadas")
    
    # Processar pares de teste
    print("\n🔄 Processando pares de TESTE...")
    
    for number in tqdm(test_pairs, desc="Processando pares de teste"):
        shadow_file = shadow_numbers[number]
        no_shadow_file = no_shadow_numbers[number]
        
        shadow_path = os.path.join(shadow_dir, shadow_file)
        no_shadow_path = os.path.join(no_shadow_dir, no_shadow_file)
        
        # Carregar imagens
        shadow_img = cv2.imread(shadow_path)
        no_shadow_img = cv2.imread(no_shadow_path)
        
        if shadow_img is None or no_shadow_img is None:
            print(f"⚠️ Erro ao carregar par {number}")
            continue
        
        print(f"📏 Processando par {number}: {shadow_img.shape[1]}x{shadow_img.shape[0]}")
        
        # Criar janelas para ambas as imagens
        shadow_windows, shadow_coords = create_sliding_windows(shadow_img, window_size, stride)
        no_shadow_windows, no_shadow_coords = create_sliding_windows(no_shadow_img, window_size, stride)
        
        # Salvar janelas com sombra (test_A)
        for i, (window, (x, y)) in enumerate(zip(shadow_windows, shadow_coords)):
            window_name = f"{number}_win_{i:04d}_x{x}_y{y}.png"
            window_path = os.path.join(test_a_dir, window_name)
            cv2.imwrite(window_path, window)
        
        # Salvar janelas sem sombra (test_C)
        for i, (window, (x, y)) in enumerate(zip(no_shadow_windows, no_shadow_coords)):
            window_name = f"{number}_win_{i:04d}_x{x}_y{y}.png"
            window_path = os.path.join(test_c_dir, window_name)
            cv2.imwrite(window_path, window)
        
        total_windows += len(shadow_windows)
        print(f"   ✅ Par {number}: {len(shadow_windows)} janelas criadas")
    
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
    
    print("\n✅ Dataset com pares criado com sucesso!")
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
    prepare_dataset_v9_pairs()
