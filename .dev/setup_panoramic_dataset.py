#!/usr/bin/env python3

import os
import shutil
import argparse
from pathlib import Path
from PIL import Image
import glob
import re
import random

def resize_panoramic_image(input_path, output_path, target_size=(320, 240)):
    """Redimensiona imagem panorâmica mantendo proporção e adicionando padding"""
    try:
        with Image.open(input_path) as img:
            # Converter para RGB se necessário
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Calcular proporção
            img_ratio = img.width / img.height
            target_ratio = target_size[0] / target_size[1]
            
            if img_ratio > target_ratio:
                # Imagem mais larga que o target - redimensionar pela largura
                new_width = target_size[0]
                new_height = int(target_size[0] / img_ratio)
            else:
                # Imagem mais alta que o target - redimensionar pela altura
                new_height = target_size[1]
                new_width = int(target_size[1] * img_ratio)
            
            # Redimensionar
            img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Criar nova imagem com padding
            new_img = Image.new('RGB', target_size, (0, 0, 0))  # Fundo preto
            
            # Calcular posição para centralizar
            x = (target_size[0] - new_width) // 2
            y = (target_size[1] - new_height) // 2
            
            # Colar imagem redimensionada
            new_img.paste(img_resized, (x, y))
            
            # Salvar
            new_img.save(output_path, 'PNG', quality=95)
            return True
            
    except Exception as e:
        print(f"❌ Erro ao processar {input_path}: {e}")
        return False

def is_valid_mission(mission_name):
    """Verifica se a pasta é uma missão válida (8 dígitos numéricos no início)"""
    if len(mission_name) < 8:
        return False
    
    # Verificar se os primeiros 8 caracteres são dígitos
    first_8 = mission_name[:8]
    return first_8.isdigit()

def setup_panoramic_dataset(base_dir, output_dir, target_size=(320, 240), max_train=None, max_test=None):
    """Configura dataset panorâmico Ladybug6"""
    
    print("🚀 Configurando dataset panorâmico...")
    print(f"📂 Diretório base: {base_dir}")
    print(f"📂 Saída: {output_dir}")
    print(f"📐 Tamanho final: {target_size[0]}x{target_size[1]} pixels")
    
    if max_train and max_test:
        print(f"📊 Máximo treino: {max_train}, Máximo teste: {max_test}")
    else:
        print(f"📊 Usando TODAS as imagens disponíveis")
    
    # Encontrar pastas de missão
    mission_dirs = []
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path) and is_valid_mission(item):
            mission_dirs.append(item)
    
    print(f"📊 Total de pastas encontradas: {len(os.listdir(base_dir))}")
    print(f"📊 Pastas válidas como missões: {len(mission_dirs)}")
    print(f"🎯 Missões válidas: {mission_dirs}")
    
    if not mission_dirs:
        print("❌ Nenhuma missão válida encontrada!")
        return False
    
    # Criar estrutura de pastas no padrão do data_manager.py
    train_C_dir = os.path.join(output_dir, "ISTD", "train", "train_C")  # Imagens sem sombra (target)
    train_A_dir = os.path.join(output_dir, "ISTD", "train", "train_A")  # Imagens com sombra (input)
    test_C_dir = os.path.join(output_dir, "ISTD", "test", "test_C")     # Imagens sem sombra (target)
    test_A_dir = os.path.join(output_dir, "ISTD", "test", "test_A")     # Imagens com sombra (input)
    
    os.makedirs(train_C_dir, exist_ok=True)
    os.makedirs(train_A_dir, exist_ok=True)
    os.makedirs(test_C_dir, exist_ok=True)
    os.makedirs(test_A_dir, exist_ok=True)
    
    # Coletar todos os pares de imagens
    all_pairs = []
    
    for mission in mission_dirs:
        mission_path = os.path.join(base_dir, mission)
        pan_dir = os.path.join(mission_path, "PAN")
        sombra_dir = os.path.join(mission_path, "PAN_SOMBRA")
        
        if not os.path.exists(pan_dir) or not os.path.exists(sombra_dir):
            print(f"⚠️ Diretórios PAN ou PAN_SOMBRA não encontrados em {mission}")
            continue
        
        # Encontrar imagens com sombra
        pan_images = glob.glob(os.path.join(pan_dir, "*.jpeg")) + glob.glob(os.path.join(pan_dir, "*.jpg"))
        print(f"📸 Missão {mission}: {len(pan_images)} imagens com sombra, ", end="")
        
        # Para cada imagem com sombra, procurar correspondente sem sombra
        for pan_img in pan_images:
            pan_name = os.path.splitext(os.path.basename(pan_img))[0]
            
            # Procurar imagem sem sombra correspondente
            sombra_candidates = [
                os.path.join(sombra_dir, f"{pan_name}.jpeg"),
                os.path.join(sombra_dir, f"{pan_name}.jpg"),
                os.path.join(sombra_dir, f"{pan_name}.png")
            ]
            
            sombra_img = None
            for candidate in sombra_candidates:
                if os.path.exists(candidate):
                    sombra_img = candidate
                    break
            
            if sombra_img:
                all_pairs.append({
                    'pan': pan_img,
                    'sombra': sombra_img,
                    'name': f"{mission}_{pan_name}",
                    'mission': mission
                })
        
        print(f"{len([p for p in all_pairs if p['mission'] == mission])} sem sombra")
    
    print(f"📊 Total de pares válidos encontrados: {len(all_pairs)}")
    
    if len(all_pairs) == 0:
        print("❌ Nenhum par de imagens válido encontrado!")
        return False
    
    # Embaralhar pares para seleção aleatória
    random.shuffle(all_pairs)
    
    # Determinar quantos pares usar
    if max_train and max_test:
        total_needed = max_train + max_test
        if len(all_pairs) > total_needed:
            selected_pairs = all_pairs[:total_needed]
            print(f"📊 Selecionando {total_needed} pares aleatórios de {len(all_pairs)} disponíveis")
        else:
            selected_pairs = all_pairs
            print(f"📊 Usando todos os {len(selected_pairs)} pares disponíveis")
        
        # Dividir em treino e teste
        train_pairs = selected_pairs[:max_train]
        test_pairs = selected_pairs[max_train:max_train + max_test]
    else:
        # Usar todas as imagens
        selected_pairs = all_pairs
        print(f"📊 Usando TODOS os {len(selected_pairs)} pares disponíveis")
        
        # Dividir 80% treino, 20% teste
        split_point = int(len(selected_pairs) * 0.8)
        train_pairs = selected_pairs[:split_point]
        test_pairs = selected_pairs[split_point:]
    
    print(f"📊 Treino: {len(train_pairs)} imagens")
    print(f"📊 Teste: {len(test_pairs)} imagens")
    
    # Processar imagens de treino
    print("🔄 Processando imagens de treino...")
    for i, pair in enumerate(train_pairs):
        if i % 100 == 0:
            print(f"  Processando {i}/{len(train_pairs)}...")
        
        # Redimensionar e salvar imagem com sombra (input) em train_A
        output_pan = os.path.join(train_A_dir, f"{pair['name']}.png")
        if not resize_panoramic_image(pair['pan'], output_pan, target_size):
            continue
        
        # Redimensionar e salvar imagem sem sombra (target) em train_C
        output_sombra = os.path.join(train_C_dir, f"{pair['name']}.png")
        if not resize_panoramic_image(pair['sombra'], output_sombra, target_size):
            continue
    
    # Processar imagens de teste
    print("🔄 Processando imagens de teste...")
    for i, pair in enumerate(test_pairs):
        if i % 100 == 0:
            print(f"  Processando {i}/{len(test_pairs)}...")
        
        # Redimensionar e salvar imagem com sombra (input) em test_A
        output_pan = os.path.join(test_A_dir, f"{pair['name']}.png")
        if not resize_panoramic_image(pair['pan'], output_pan, target_size):
            continue
        
        # Redimensionar e salvar imagem sem sombra (target) em test_C
        output_sombra = os.path.join(test_C_dir, f"{pair['name']}.png")
        if not resize_panoramic_image(pair['sombra'], output_sombra, target_size):
            continue
    
    # Verificar resultados
    train_A_count = len([f for f in os.listdir(train_A_dir) if f.endswith('.png')])
    train_C_count = len([f for f in os.listdir(train_C_dir) if f.endswith('.png')])
    test_A_count = len([f for f in os.listdir(test_A_dir) if f.endswith('.png')])
    test_C_count = len([f for f in os.listdir(test_C_dir) if f.endswith('.png')])
    
    print("✅ Dataset configurado com sucesso!")
    print(f"📊 Treino A (com sombra): {train_A_count} imagens")
    print(f"📊 Treino C (sem sombra): {train_C_count} imagens")
    print(f"📊 Teste A (com sombra): {test_A_count} imagens")
    print(f"📊 Teste C (sem sombra): {test_C_count} imagens")
    
    return True

def create_panoramic_config(output_dir, num_epochs=100, batch_size=1):
    """Cria arquivo de configuração para o dataset panorâmico"""
    
    # Contar imagens de treino
    train_A_dir = os.path.join(output_dir, "ISTD", "train", "train_A")
    train_count = len([f for f in os.listdir(train_A_dir) if f.endswith('.png')]) if os.path.exists(train_A_dir) else 0
    
    config_content = f"""# Configuração para dataset panorâmico Ladybug6 (otimizada para GPU menor)
datasets_dir: {output_dir}/ISTD/train
valset_dir: {output_dir}/ISTD/test
train_list: train_list.txt
test_list:
validation_list: val_list.txt
out_dir: ./results_panoramic

cuda: True
gpu_ids: [0]

train_size: 2
val_size: 0
batchsize: {batch_size}
validation_batchsize: {batch_size}
epoch: {num_epochs}
n_data: {train_count}  # Dataset completo
width: 320
height: 240
threads: 2

lr: 0.0004
beta1: 0.5
lamb: 100
minimax: 1

gen_init:
dis_init:
in_ch: 3
out_ch: 3

manualSeed: 0
snapshot_interval: 1
"""
    
    config_path = os.path.join(output_dir, "config_panoramic.yml")
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(config_content)
    
    print(f"📄 Configuração salva: {config_path}")
    return config_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Configurar dataset panorâmico Ladybug6')
    parser.add_argument('--base_dir', type=str, required=True, help='Diretório base com as missões')
    parser.add_argument('--output_dir', type=str, default='./data_panoramic', help='Diretório de saída')
    parser.add_argument('--width', type=int, default=320, help='Largura final')
    parser.add_argument('--height', type=int, default=240, help='Altura final')
    parser.add_argument('--max_train', type=int, help='Máximo de imagens de treino (opcional)')
    parser.add_argument('--max_test', type=int, help='Máximo de imagens de teste (opcional)')
    parser.add_argument('--epochs', type=int, default=100, help='Número de épocas')
    parser.add_argument('--batch_size', type=int, default=1, help='Tamanho do batch')
    args = parser.parse_args()
    
    print("🚀 Configurando dataset panorâmico Ladybug6")
    print(f"📂 Base: {args.base_dir}")
    print(f"📂 Saída: {args.output_dir}")
    print(f"📐 Tamanho: {args.width}x{args.height}")
    
    if args.max_train and args.max_test:
        print(f"📊 Dataset pequeno: {args.max_train} treino + {args.max_test} teste")
    else:
        print(f"📊 Dataset completo: TODAS as imagens disponíveis")
    
    if setup_panoramic_dataset(args.base_dir, args.output_dir, (args.width, args.height), args.max_train, args.max_test):
        create_panoramic_config(args.output_dir, args.epochs, args.batch_size)
        print("🎉 Dataset configurado com sucesso!")
    else:
        print("❌ Falha na configuração do dataset") 