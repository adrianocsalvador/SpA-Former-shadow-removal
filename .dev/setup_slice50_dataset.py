#!/usr/bin/env python3

import os
import cv2
import numpy as np
from PIL import Image
import argparse
import shutil
from pathlib import Path

def remove_black_borders(image, black_threshold=10):
    """Remove bordas pretas da imagem (horizontal e vertical)"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Encontrar coordenadas não-pretas
    coords = cv2.findNonZero(gray)
    x, y, w, h = cv2.boundingRect(coords)
    
    # Cortar a imagem
    cropped = image[y:y+h, x:x+w]
    
    # Adicionar margem de segurança (evitar cortar muito)
    margin = 50
    height, width = cropped.shape[:2]
    start_y = max(0, margin)
    end_y = min(height, height - margin)
    start_x = max(0, margin)
    end_x = min(width, width - margin)
    
    final_cropped = cropped[start_y:end_y, start_x:end_x]
    
    return final_cropped, (x, y, w, h)

def split_image_into_slices(image, slice_width=50, overlap=5, max_height=1000):
    """Divide a imagem em fatias verticais com sobreposição e altura limitada"""
    height, width = image.shape[:2]
    
    # Limitar a altura se necessário
    if height > max_height:
        # Cortar do centro para manter a parte mais importante
        start_y = (height - max_height) // 2
        end_y = start_y + max_height
        image = image[start_y:end_y, :]
        height = max_height
        print(f"    ✂️ Altura limitada: {max_height}px (cortada do centro)")
    
    effective_width = slice_width - overlap
    num_slices = int(np.ceil((width - slice_width) / effective_width)) + 1
    slices = []
    positions = []
    
    for i in range(num_slices):
        start_x = i * effective_width
        end_x = min(start_x + slice_width, width)
        if end_x == width and start_x + slice_width > width:
            start_x = width - slice_width
        slice_img = image[:, start_x:end_x]
        slices.append(slice_img)
        positions.append((start_x, end_x))
        if end_x >= width:
            break
    
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
    
    # Colocar imagem redimensionada no centro
    final_img[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized
    
    return final_img

def setup_slice50_dataset(base_dir, output_dir, target_size=(640, 480), max_images=10, slice_width=50, overlap=5):
    """Configura dataset com fatias de 50px de largura"""
    
    # Verificar se target_size é uma string no formato "50x5073"
    if isinstance(target_size, str) and 'x' in target_size:
        try:
            width_str, height_str = target_size.split('x')
            target_size = (int(width_str), int(height_str))
            skip_resize = True
            print(f"🎯 Modo sem redimensionamento: {target_size[0]}x{target_size[1]}")
        except:
            print(f"⚠️ Formato inválido: {target_size}, usando redimensionamento padrão")
            skip_resize = False
    else:
        skip_resize = False
    
    print(f"🚀 Configurando dataset com fatias de {slice_width}px...")
    print(f"📊 Tamanho alvo: {target_size}")
    print(f"📊 Máximo de imagens: {max_images}")
    print(f"🔪 Largura da fatia: {slice_width}px")
    print(f"🔄 Sobreposição: {overlap}px")
    print(f"🔄 Redimensionamento: {'Não' if skip_resize else 'Sim'}")
    
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
        
        # Limitar número de imagens
        pan_images = pan_images[:max_images]
        
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
            
            # Remover bordas pretas (155 linhas de baixo)
            pan_cropped, pan_crop_info = remove_black_borders(pan_img)
            sombra_cropped, sombra_crop_info = remove_black_borders(sombra_img)
            
            print(f"    ✂️ Após remoção de bordas: {pan_cropped.shape} e {sombra_cropped.shape}")
            
            # Dividir em fatias (limitando altura para economizar memória)
            pan_slices, pan_positions = split_image_into_slices(pan_cropped, slice_width, overlap, max_height=800)
            sombra_slices, sombra_positions = split_image_into_slices(sombra_cropped, slice_width, overlap, max_height=800)
            
            print(f"    🔪 Criadas {len(pan_slices)} fatias")
            
            # Processar cada fatia
            for i, (pan_slice, sombra_slice) in enumerate(zip(pan_slices, sombra_slices)):
                # Redimensionar mantendo aspect ratio
                if not skip_resize:
                    pan_resized = resize_with_aspect_ratio(pan_slice, target_size)
                    sombra_resized = resize_with_aspect_ratio(sombra_slice, target_size)
                else:
                    pan_resized = pan_slice
                    sombra_resized = sombra_slice
                
                print(f"    📏 Fatia {i+1}: {pan_slice.shape} → {pan_resized.shape}")
                
                # Nome do arquivo
                output_name = f"{base_name}_slice{i+1:03d}.png"
                
                # Decidir se vai para treino ou teste (80/20)
                if np.random.random() < 0.8:
                    # Treino
                    cv2.imwrite(os.path.join(train_A_dir, output_name), sombra_resized)
                    cv2.imwrite(os.path.join(train_C_dir, output_name), pan_resized)
                    train_count += 1
                else:
                    # Teste
                    cv2.imwrite(os.path.join(test_A_dir, output_name), sombra_resized)
                    cv2.imwrite(os.path.join(test_C_dir, output_name), pan_resized)
                    test_count += 1
                
                total_pairs += 1
            
            if total_pairs >= max_images * 10:  # Limitar total de fatias
                break
        
        if total_pairs >= max_images * 10:
            break
    
    print(f"✅ Dataset configurado com sucesso!")
    print(f"📊 Total de pares: {total_pairs}")
    print(f"📊 Treino: {train_count}")
    print(f"📊 Teste: {test_count}")
    print(f"📁 Saída: {output_dir}")
    
    # Criar arquivo de configuração
    config_content = f"""# Configuração para dataset com fatias de {slice_width}px
datasets_dir: {output_dir}/ISTD/train
valset_dir: {output_dir}/ISTD/test
train_list: train_list.txt
test_list:
validation_list: val_list.txt
out_dir: ./results_slice50

cuda: True  # Usar GPU agora que temos resolução reduzida
gpu_ids: [0]  # Usar GPU 0

train_size: 2
val_size: 0
batchsize: 2  # Batch size otimizado para fatias grandes
validation_batchsize: 2
epoch: 100
n_data: {total_pairs}
width: {target_size[0]}
height: {target_size[1]}
threads: 4

lr: 0.0002  # Learning rate reduzido para estabilidade
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
    
    config_path = os.path.join(output_dir, 'config_slice50.yml')
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    print(f"📝 Configuração salva: {config_path}")
    
    return output_dir

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Configurar dataset com fatias de 50px')
    parser.add_argument('--base_dir', type=str, required=True,
                       help='Diretório base com as missões')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Diretório de saída')
    parser.add_argument('--target_size', type=str, default='512,384',
                       help='Tamanho alvo (largura,altura)')
    parser.add_argument('--max_images', type=int, default=10,
                       help='Número máximo de imagens para processar')
    parser.add_argument('--slice_width', type=int, default=50,
                       help='Largura da fatia (padrão: 50)')
    parser.add_argument('--overlap', type=int, default=5,
                       help='Sobreposição entre fatias (padrão: 5)')
    
    args = parser.parse_args()
    
    # Parse target size (aceita tanto vírgula quanto 'x')
    if 'x' in args.target_size:
        target_size = args.target_size  # Passar como string para processamento interno
    else:
        target_size = tuple(map(int, args.target_size.split(',')))
    
    print("🚀 Configuração de Dataset com Fatias de 50px")
    print("=" * 50)
    
    # Configurar dataset
    output_dir = setup_slice50_dataset(
        args.base_dir, 
        args.output_dir, 
        target_size, 
        args.max_images,
        args.slice_width,
        args.overlap
    )
    
    print("\n🎉 Configuração concluída!")
    print(f"📁 Dataset: {output_dir}")
    print(f"⚙️ Config: {output_dir}/config_slice50.yml")
    
    print(f"\n💡 Para treinar com fatias de {args.slice_width}px:")
    print(f"python3 .dev/start_high_res_training.py --config {output_dir}/config_slice50.yml") 