#!/usr/bin/env python3

import os
import cv2
import numpy as np
from PIL import Image
import argparse
import shutil
from pathlib import Path

def remove_black_borders(image):
    """Remove bordas pretas da imagem"""
    # Converter para grayscale para análise
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Encontrar coordenadas não-zero (não pretas)
    coords = cv2.findNonZero(gray)
    x, y, w, h = cv2.boundingRect(coords)
    
    # Cortar a imagem removendo as bordas pretas
    cropped = image[y:y+h, x:x+w]
    
    return cropped, (x, y, w, h)

def split_image_vertically(image, num_slices=3, overlap=0.1):
    """Divide a imagem em fatias verticais com sobreposição"""
    height, width = image.shape[:2]
    
    # Calcular tamanho de cada fatia
    slice_width = width // num_slices
    overlap_pixels = int(slice_width * overlap)
    
    slices = []
    positions = []
    
    for i in range(num_slices):
        # Calcular posições de início e fim
        start_x = i * slice_width
        end_x = start_x + slice_width
        
        # Adicionar sobreposição (exceto para a primeira fatia)
        if i > 0:
            start_x -= overlap_pixels
        
        # Garantir que não ultrapasse os limites
        start_x = max(0, start_x)
        end_x = min(width, end_x)
        
        # Extrair fatia
        slice_img = image[:, start_x:end_x]
        slices.append(slice_img)
        positions.append((start_x, end_x))
    
    return slices, positions

def resize_with_aspect_ratio(image, target_size, fill_color=(0, 0, 0)):
    """Redimensiona mantendo aspect ratio e preenche com cor"""
    target_width, target_height = target_size
    
    # Calcular aspect ratios
    img_height, img_width = image.shape[:2]
    aspect_ratio = img_width / img_height
    target_aspect = target_width / target_height
    
    if aspect_ratio > target_aspect:
        # Imagem mais larga - redimensionar por largura
        new_width = target_width
        new_height = int(target_width / aspect_ratio)
    else:
        # Imagem mais alta - redimensionar por altura
        new_height = target_height
        new_width = int(target_height * aspect_ratio)
    
    # Redimensionar
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
    
    # Criar imagem final com padding
    final_img = np.full((target_height, target_width, 3), fill_color, dtype=np.uint8)
    
    # Calcular posição para centralizar
    y_offset = (target_height - new_height) // 2
    x_offset = (target_width - new_width) // 2
    
    # Colar imagem redimensionada
    final_img[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized
    
    return final_img

def setup_high_res_dataset(base_dir, output_dir, target_size=(1280, 960), num_slices=3, overlap=0.1):
    """Configura dataset de alta resolução com fatias verticais"""
    
    print(f"🚀 Configurando dataset de alta resolução...")
    print(f"📊 Tamanho alvo: {target_size}")
    print(f"🔪 Número de fatias: {num_slices}")
    print(f"📐 Sobreposição: {overlap*100}%")
    
    # Criar diretórios
    train_A_dir = os.path.join(output_dir, 'ISTD', 'train', 'train_A')
    train_C_dir = os.path.join(output_dir, 'ISTD', 'train', 'train_C')
    test_A_dir = os.path.join(output_dir, 'ISTD', 'test', 'test_A')
    test_C_dir = os.path.join(output_dir, 'ISTD', 'test', 'test_C')
    
    for dir_path in [train_A_dir, train_C_dir, test_A_dir, test_C_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    # Encontrar missões
    mission_dirs = []
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path) and len(item) >= 8:
            try:
                int(item[:8])  # Verificar se os primeiros 8 caracteres são números
                mission_dirs.append(item)
            except ValueError:
                continue
    
    print(f"📁 Missões encontradas: {len(mission_dirs)}")
    
    total_pairs = 0
    train_count = 0
    test_count = 0
    
    for mission in mission_dirs:
        print(f"🔄 Processando missão: {mission}")
        
        pan_dir = os.path.join(base_dir, mission, 'PAN')
        pan_sombra_dir = os.path.join(base_dir, mission, 'PAN_SOMBRA')
        
        if not os.path.exists(pan_dir) or not os.path.exists(pan_sombra_dir):
            print(f"⚠️ Diretórios não encontrados para missão {mission}")
            continue
        
        # Listar imagens
        pan_images = [f for f in os.listdir(pan_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        print(f"    📸 Imagens encontradas em PAN: {len(pan_images)}")
        if len(pan_images) > 0:
            print(f"    📸 Primeiras imagens: {pan_images[:3]}")
        
        for img_name in pan_images:
            pan_path = os.path.join(pan_dir, img_name)
            
            # Tentar encontrar imagem correspondente com extensões diferentes
            base_name = os.path.splitext(img_name)[0]
            possible_extensions = ['.jpg', '.jpeg', '.png']
            
            sombra_path = None
            for ext in possible_extensions:
                test_path = os.path.join(pan_sombra_dir, base_name + ext)
                if os.path.exists(test_path):
                    sombra_path = test_path
                    break
            
            print(f"    🔍 Processando: {img_name}")
            print(f"    📁 PAN: {pan_path}")
            print(f"    📁 SOMBRA: {sombra_path}")
            
            if sombra_path is None:
                print(f"    ⚠️ Imagem sombra não encontrada para: {img_name}")
                continue
            
            # Carregar imagens
            pan_img = cv2.imread(pan_path)
            sombra_img = cv2.imread(sombra_path)
            
            if pan_img is None or sombra_img is None:
                print(f"    ❌ Erro ao carregar imagens")
                continue
            
            print(f"    ✅ Imagens carregadas: {pan_img.shape} e {sombra_img.shape}")
            
            # Remover bordas pretas
            pan_cropped, pan_coords = remove_black_borders(pan_img)
            sombra_cropped, sombra_coords = remove_black_borders(sombra_img)
            
            print(f"    ✂️ Após remoção de bordas: {pan_cropped.shape} e {sombra_cropped.shape}")
            
            # Dividir em fatias
            pan_slices, pan_positions = split_image_vertically(pan_cropped, num_slices, overlap)
            sombra_slices, sombra_positions = split_image_vertically(sombra_cropped, num_slices, overlap)
            
            print(f"    🔪 Fatias criadas: {len(pan_slices)}")
            
            # Processar cada fatia
            for i, (pan_slice, sombra_slice) in enumerate(zip(pan_slices, sombra_slices)):
                # Redimensionar mantendo aspect ratio
                pan_resized = resize_with_aspect_ratio(pan_slice, target_size)
                sombra_resized = resize_with_aspect_ratio(sombra_slice, target_size)
                
                # Nome do arquivo
                base_name = os.path.splitext(img_name)[0]
                slice_name = f"{base_name}_slice_{i:02d}.png"
                
                # Decidir se vai para treino ou teste (80/20)
                if np.random.random() < 0.8:
                    # Treino
                    cv2.imwrite(os.path.join(train_A_dir, slice_name), sombra_resized)
                    cv2.imwrite(os.path.join(train_C_dir, slice_name), pan_resized)
                    train_count += 1
                else:
                    # Teste
                    cv2.imwrite(os.path.join(test_A_dir, slice_name), sombra_resized)
                    cv2.imwrite(os.path.join(test_C_dir, slice_name), pan_resized)
                    test_count += 1
                
                total_pairs += 1
                
                if total_pairs % 10 == 0:
                    print(f"  Processados: {total_pairs} pares")
    
    print(f"✅ Dataset configurado com sucesso!")
    print(f"📊 Total de pares: {total_pairs}")
    print(f"📊 Treino: {train_count}")
    print(f"📊 Teste: {test_count}")
    print(f"📁 Saída: {output_dir}")
    
    return output_dir

def create_high_res_config(output_dir, target_size=(1280, 960)):
    """Cria arquivo de configuração para alta resolução"""
    
    config_content = f"""# Configuração para dataset de alta resolução
datasets_dir: {output_dir}/ISTD/train
valset_dir: {output_dir}/ISTD/test
train_list: train_list.txt
test_list:
validation_list: val_list.txt
out_dir: ./results_high_res

cuda: True
gpu_ids: [0]

train_size: 2
val_size: 0
batchsize: 1  # Reduzido para alta resolução
validation_batchsize: 1
epoch: 100
n_data: 1000  # Será atualizado automaticamente
width: {target_size[0]}
height: {target_size[1]}
threads: 2

lr: 0.0002  # Learning rate reduzido para alta resolução
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
    
    config_path = os.path.join(output_dir, 'config_high_res.yml')
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    print(f"📝 Configuração salva: {config_path}")
    return config_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Configurar dataset de alta resolução')
    parser.add_argument('--base_dir', type=str, required=True, 
                       help='Diretório base com as missões')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Diretório de saída')
    parser.add_argument('--target_size', type=str, default='512,384',
                       help='Tamanho alvo (largura,altura)')
    parser.add_argument('--num_slices', type=int, default=1,
                       help='Número de fatias verticais')
    parser.add_argument('--overlap', type=float, default=0.1,
                       help='Sobreposição entre fatias (0.0-1.0)')
    
    args = parser.parse_args()
    
    # Parse target size
    target_size = tuple(map(int, args.target_size.split(',')))
    
    print("🚀 Configuração de Dataset de Alta Resolução")
    print("=" * 50)
    
    # Configurar dataset
    output_dir = setup_high_res_dataset(
        args.base_dir, 
        args.output_dir, 
        target_size, 
        args.num_slices, 
        args.overlap
    )
    
    # Criar configuração
    config_path = create_high_res_config(output_dir, target_size)
    
    print(f"\n🎉 Configuração concluída!")
    print(f"📁 Dataset: {output_dir}")
    print(f"⚙️ Config: {config_path}")
    print(f"\n💡 Para treinar com alta resolução:")
    print(f"python3 .dev/real_train.py --config {config_path}") 