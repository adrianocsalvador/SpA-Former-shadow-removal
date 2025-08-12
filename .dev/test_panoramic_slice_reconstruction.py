#!/usr/bin/env python3
"""
Script para testar processamento de imagem panorâmica em fatias
- Reduz resolução pela metade
- Corta em fatias de 50px
- Processa cada fatia
- Reconstrói na resolução original
"""

import os
import sys
import cv2
import torch
import numpy as np
import argparse
from PIL import Image
import matplotlib.pyplot as plt

# Adicionar o diretório raiz ao path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from SpA_Former import Generator
from utils import gpu_manage

def remove_black_borders(image, black_threshold=10):
    """Remove bordas pretas da imagem"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    coords = cv2.findNonZero(gray)
    x, y, w, h = cv2.boundingRect(coords)
    cropped = image[y:y+h, x:x+w]
    return cropped, (x, y, w, h)

def split_image_into_slices(image, slice_width=50, overlap=5):
    """Divide a imagem em fatias verticais com sobreposição"""
    height, width = image.shape[:2]
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

def reconstruct_image_from_slices(processed_slices, positions, target_height, target_width, slice_width=50, overlap=5, crop_info=None, is_attention=False):
    """Reconstrói a imagem a partir das fatias processadas com normalização de sobreposição"""
    # Criar imagem vazia com dimensões do target (imagem reduzida)
    if is_attention:
        # Para attention maps (2D)
        reconstructed = np.zeros((target_height, target_width), dtype=np.uint8)
        overlap_count = np.zeros((target_height, target_width), dtype=np.float32)
    else:
        # Para imagens RGB (3D)
        reconstructed = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        overlap_count = np.zeros((target_height, target_width, 3), dtype=np.float32)
    
    for i, (slice_img, (start_x, end_x)) in enumerate(zip(processed_slices, positions)):
        # Redimensionar fatia para altura do target
        slice_height, slice_width_actual = slice_img.shape[:2]
        slice_resized = cv2.resize(slice_img, (slice_width_actual, target_height), interpolation=cv2.INTER_CUBIC)
        
        # Posições na imagem target (reduzida)
        target_start_x = start_x
        target_end_x = end_x
        target_start_y = 0
        target_end_y = target_height
        
        # Adicionar à imagem reconstruída com blending normalizado
        if i == 0:  # Primeira fatia
            reconstructed[target_start_y:target_end_y, target_start_x:target_end_x] = slice_resized
            overlap_count[target_start_y:target_end_y, target_start_x:target_end_x] = 1.0
        else:
            # Para fatias subsequentes, fazer blending normalizado na região de sobreposição
            overlap_start = max(target_start_x, target_start_x)
            overlap_end = min(target_end_x, target_start_x + overlap)
            
            if overlap_start < overlap_end:
                # Região de sobreposição - normalizar com pesos
                overlap_width = overlap_end - overlap_start
                slice_overlap_start = overlap_start - target_start_x
                
                # Criar pesos para blending suave
                weights = np.linspace(0, 1, overlap_width)
                if is_attention:
                    weights = np.tile(weights[:, np.newaxis], (1, target_height))
                else:
                    weights = np.tile(weights[:, np.newaxis, np.newaxis], (1, target_height, 3))
                
                # Aplicar blending normalizado
                for j, x in enumerate(range(overlap_start, overlap_end)):
                    slice_j = slice_overlap_start + j
                    if 0 <= slice_j < slice_resized.shape[1]:
                        weight = weights[j]
                        reconstructed[target_start_y:target_end_y, x] = (
                            (1 - weight) * reconstructed[target_start_y:target_end_y, x] + 
                            weight * slice_resized[:, slice_j]
                        )
                        overlap_count[target_start_y:target_end_y, x] = 1.0
                
                # Região sem sobreposição
                if overlap_end < target_end_x:
                    slice_start = slice_overlap_start + overlap_width
                    reconstructed[target_start_y:target_end_y, overlap_end:target_end_x] = slice_resized[:, slice_start:]
                    overlap_count[target_start_y:target_end_y, overlap_end:target_end_x] = 1.0
            else:
                # Sem sobreposição
                reconstructed[target_start_y:target_end_y, target_start_x:target_end_x] = slice_resized
                overlap_count[target_start_y:target_end_y, target_start_x:target_end_x] = 1.0
    
    # Normalizar regiões com sobreposição
    overlap_mask = overlap_count > 0
    reconstructed[overlap_mask] = reconstructed[overlap_mask] / np.maximum(overlap_count[overlap_mask], 1.0)
    
    return reconstructed.astype(np.uint8)

def restore_black_borders(reconstructed_img, original_img, crop_info):
    """Restaura os pixels pretos removidos na imagem final"""
    original_height, original_width = original_img.shape[:2]
    crop_x, crop_y, crop_w, crop_h = crop_info
    
    # Criar imagem final com dimensões originais
    final_img = np.zeros((original_height, original_width, 3), dtype=np.uint8)
    
    # Redimensionar a imagem reconstruída para as dimensões da região cortada
    reconstructed_resized = cv2.resize(reconstructed_img, (crop_w, crop_h), interpolation=cv2.INTER_LANCZOS4)
    
    # Copiar a região processada
    final_img[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w] = reconstructed_resized
    
    return final_img

def restore_black_borders_attention(reconstructed_img, original_img, crop_info):
    """Restaura os pixels pretos removidos na imagem final (para attention maps)"""
    original_height, original_width = original_img.shape[:2]
    crop_x, crop_y, crop_w, crop_h = crop_info
    
    # Criar imagem final com dimensões originais (2D para attention)
    final_img = np.zeros((original_height, original_width), dtype=np.uint8)
    
    # Redimensionar a imagem reconstruída para as dimensões da região cortada
    reconstructed_resized = cv2.resize(reconstructed_img, (crop_w, crop_h), interpolation=cv2.INTER_LANCZOS4)
    
    # Copiar a região processada
    final_img[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w] = reconstructed_resized
    
    return final_img

def process_slice_with_model(slice_img, model, device):
    """Processa uma fatia com o modelo e retorna tanto o output quanto o attention"""
    # Converter para tensor
    slice_tensor = torch.from_numpy(slice_img).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    slice_tensor = slice_tensor.to(device)
    
    # Processar com modelo
    with torch.no_grad():
        attention, output = model(slice_tensor)
    
    # Converter de volta para numpy
    output = output.squeeze().permute(1, 2, 0).cpu().numpy()
    output = np.clip(output * 255, 0, 255).astype(np.uint8)
    
    # Processar attention map
    attention = attention.squeeze().cpu().numpy()
    attention = np.clip(attention * 255, 0, 255).astype(np.uint8)
    
    return output, attention

def main():
    parser = argparse.ArgumentParser(description='Processar imagem panorâmica em fatias')
    parser.add_argument('--input', type=str, required=True, help='Caminho da imagem de entrada')
    parser.add_argument('--model', type=str, required=True, help='Caminho do modelo')
    parser.add_argument('--output_dir', type=str, default='./out', help='Diretório de saída')
    parser.add_argument('--slice_width', type=int, default=50, help='Largura das fatias')
    parser.add_argument('--overlap', type=int, default=5, help='Sobreposição entre fatias')
    
    args = parser.parse_args()
    
    # Criar diretório de saída
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Configurar GPU/CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Usando dispositivo: {device}")
    
    # Carregar modelo
    print(f"📦 Carregando modelo: {args.model}")
    model = Generator(gpu_ids=[0] if torch.cuda.is_available() else [])
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()
    model.to(device)
    
    # Carregar imagem original
    print(f"🖼️ Carregando imagem: {args.input}")
    original_img = cv2.imread(args.input)
    if original_img is None:
        raise ValueError(f"Não foi possível carregar a imagem: {args.input}")
    
    original_height, original_width = original_img.shape[:2]
    print(f"📏 Dimensões originais: {original_width}x{original_height}")
    
    # Remover bordas pretas
    print("✂️ Removendo bordas pretas...")
    cropped_img, crop_info = remove_black_borders(original_img)
    cropped_height, cropped_width = cropped_img.shape[:2]
    print(f"📏 Após remoção de bordas: {cropped_width}x{cropped_height}")
    print(f"📏 Informações de crop: x={crop_info[0]}, y={crop_info[1]}, w={crop_info[2]}, h={crop_info[3]}")
    
    # Reduzir resolução pela metade
    print("📐 Reduzindo resolução pela metade...")
    half_width = cropped_width // 2
    half_height = cropped_height // 2
    half_img = cv2.resize(cropped_img, (half_width, half_height), interpolation=cv2.INTER_LANCZOS4)
    print(f"📏 Resolução reduzida: {half_width}x{half_height}")
    
    # Salvar imagem reduzida
    half_img_path = os.path.join(args.output_dir, '01_half_resolution.png')
    cv2.imwrite(half_img_path, half_img)
    print(f"💾 Imagem reduzida salva: {half_img_path}")
    
    # Dividir em fatias
    print(f"🔪 Dividindo em fatias de {args.slice_width}px...")
    slices, positions = split_image_into_slices(half_img, args.slice_width, args.overlap)
    print(f"📊 Total de fatias: {len(slices)}")
    
    # Salvar fatias originais
    slices_dir = os.path.join(args.output_dir, 'slices_original')
    os.makedirs(slices_dir, exist_ok=True)
    for i, (slice_img, pos) in enumerate(zip(slices, positions)):
        slice_path = os.path.join(slices_dir, f'slice_{i:03d}_{pos[0]}-{pos[1]}.png')
        cv2.imwrite(slice_path, slice_img)
    
    # Processar cada fatia
    print("🔄 Processando fatias...")
    processed_slices = []
    attention_maps = []
    for i, (slice_img, pos) in enumerate(zip(slices, positions)):
        print(f"  Processando fatia {i+1}/{len(slices)} ({pos[0]}-{pos[1]})")
        
        # Processar com modelo
        processed_slice, attention_map = process_slice_with_model(slice_img, model, device)
        processed_slices.append(processed_slice)
        attention_maps.append(attention_map)
        
        # Salvar fatia processada
        slice_path = os.path.join(slices_dir, f'processed_slice_{i:03d}_{pos[0]}-{pos[1]}.png')
        cv2.imwrite(slice_path, processed_slice)
        
        # Salvar attention map
        attention_path = os.path.join(slices_dir, f'attention_slice_{i:03d}_{pos[0]}-{pos[1]}.png')
        cv2.imwrite(attention_path, attention_map)
    
    # Reconstruir imagem
    print("🔧 Reconstruindo imagem...")
    reconstructed_half = reconstruct_image_from_slices(
        processed_slices, positions, half_height, half_width, args.slice_width, args.overlap, crop_info
    )
    
    # Reconstruir attention map
    print("🔧 Reconstruindo attention map...")
    reconstructed_attention_half = reconstruct_image_from_slices(
        attention_maps, positions, half_height, half_width, args.slice_width, args.overlap, crop_info, is_attention=True
    )
    
    # Salvar imagem reconstruída (resolução reduzida)
    reconstructed_half_path = os.path.join(args.output_dir, '02_reconstructed_half.png')
    cv2.imwrite(reconstructed_half_path, reconstructed_half)
    print(f"💾 Imagem reconstruída (meia resolução) salva: {reconstructed_half_path}")
    
    # Salvar attention map reconstruído (resolução reduzida)
    attention_half_path = os.path.join(args.output_dir, '02_attention_half.png')
    cv2.imwrite(attention_half_path, reconstructed_attention_half)
    print(f"💾 Attention map (meia resolução) salvo: {attention_half_path}")
    
    # Redimensionar para resolução original
    print("📐 Redimensionando para resolução original...")
    reconstructed_full = cv2.resize(reconstructed_half, (original_width, original_height), interpolation=cv2.INTER_LANCZOS4)
    attention_full = cv2.resize(reconstructed_attention_half, (original_width, original_height), interpolation=cv2.INTER_LANCZOS4)
    
    # Restaurar pixels pretos na imagem final
    print("🔧 Restaurando pixels pretos...")
    reconstructed_with_borders = restore_black_borders(reconstructed_full, original_img, crop_info)
    attention_with_borders = restore_black_borders_attention(attention_full, original_img, crop_info)
    
    # Salvar imagem final
    reconstructed_full_path = os.path.join(args.output_dir, '03_reconstructed_full.png')
    cv2.imwrite(reconstructed_full_path, reconstructed_with_borders)
    print(f"💾 Imagem final salva: {reconstructed_full_path}")
    
    # Salvar attention map final
    attention_full_path = os.path.join(args.output_dir, '03_attention_full.png')
    cv2.imwrite(attention_full_path, attention_with_borders)
    print(f"💾 Attention map final salvo: {attention_full_path}")
    
    # Criar overlay do attention map
    print("🖼️ Criando overlay do attention map...")
    attention_colored = cv2.applyColorMap(attention_with_borders, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(original_img, 0.7, attention_colored, 0.3, 0)
    overlay_path = os.path.join(args.output_dir, '03_attention_overlay.png')
    cv2.imwrite(overlay_path, overlay)
    print(f"💾 Overlay do attention map salvo: {overlay_path}")
    
    # Criar comparação
    print("🖼️ Criando comparação...")
    comparison = np.hstack([original_img, reconstructed_with_borders])
    comparison_path = os.path.join(args.output_dir, '04_comparison.png')
    cv2.imwrite(comparison_path, comparison)
    print(f"💾 Comparação salva: {comparison_path}")
    
    print("✅ Processamento concluído!")
    print(f"📁 Resultados salvos em: {args.output_dir}")

if __name__ == "__main__":
    main() 