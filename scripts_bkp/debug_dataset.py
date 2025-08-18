#!/usr/bin/env python3
"""
Script de debug para investigar o problema com o carregamento do dataset
"""

import os
import cv2
import numpy as np
import torch
from PIL import Image

def debug_dataset():
    """Debug do dataset"""
    data_dir = "/mnt/48EC7EE9EC7ED0A4/Teste_Sombra/dataset_v9_pairs_512x512/train"
    shadow_dir = os.path.join(data_dir, 'train_A')
    shadow_free_dir = os.path.join(data_dir, 'train_C')
    
    print(f"🔍 Debug do dataset em: {data_dir}")
    print(f"📁 Shadow dir: {shadow_dir}")
    print(f"📁 Shadow-free dir: {shadow_free_dir}")
    
    # Verificar se os diretórios existem
    if not os.path.exists(shadow_dir):
        print(f"❌ Diretório shadow não existe: {shadow_dir}")
        return
    if not os.path.exists(shadow_free_dir):
        print(f"❌ Diretório shadow-free não existe: {shadow_free_dir}")
        return
    
    # Listar arquivos
    shadow_files = [f for f in os.listdir(shadow_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    shadow_free_files = [f for f in os.listdir(shadow_free_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"📊 Total de arquivos shadow: {len(shadow_files)}")
    print(f"📊 Total de arquivos shadow-free: {len(shadow_free_files)}")
    
    # Filtrar por tamanho
    shadow_files_valid = []
    shadow_free_files_valid = []
    
    for f in shadow_files:
        file_path = os.path.join(shadow_dir, f)
        if os.path.getsize(file_path) > 10000:  # > 10KB
            shadow_files_valid.append(f)
    
    for f in shadow_free_files:
        file_path = os.path.join(shadow_free_dir, f)
        if os.path.getsize(file_path) > 10000:  # > 10KB
            shadow_free_files_valid.append(f)
    
    print(f"📊 Arquivos shadow válidos (>10KB): {len(shadow_files_valid)}")
    print(f"📊 Arquivos shadow-free válidos (>10KB): {len(shadow_free_files_valid)}")
    
    # Testar carregamento de algumas imagens
    print("\n🧪 Testando carregamento de imagens...")
    
    for i in range(min(5, len(shadow_files_valid))):
        shadow_file = shadow_files_valid[i]
        shadow_free_file = shadow_file  # Mesmo nome
        
        if shadow_free_file not in shadow_free_files_valid:
            print(f"⚠️ Arquivo correspondente não encontrado: {shadow_free_file}")
            continue
        
        shadow_path = os.path.join(shadow_dir, shadow_file)
        shadow_free_path = os.path.join(shadow_free_dir, shadow_free_file)
        
        print(f"\n📸 Testando {shadow_file}:")
        print(f"   Shadow path: {shadow_path}")
        print(f"   Shadow-free path: {shadow_free_path}")
        
        # Testar com cv2
        try:
            shadow_img_cv2 = cv2.imread(shadow_path)
            shadow_free_img_cv2 = cv2.imread(shadow_free_path)
            
            if shadow_img_cv2 is None:
                print(f"   ❌ cv2.imread falhou para shadow")
            else:
                print(f"   ✅ cv2.imread shadow: {shadow_img_cv2.shape}")
            
            if shadow_free_img_cv2 is None:
                print(f"   ❌ cv2.imread falhou para shadow-free")
            else:
                print(f"   ✅ cv2.imread shadow-free: {shadow_free_img_cv2.shape}")
                
        except Exception as e:
            print(f"   ❌ Erro cv2: {e}")
        
        # Testar com PIL
        try:
            shadow_img_pil = Image.open(shadow_path)
            shadow_free_img_pil = Image.open(shadow_free_path)
            
            print(f"   ✅ PIL shadow: {shadow_img_pil.size}, mode: {shadow_img_pil.mode}")
            print(f"   ✅ PIL shadow-free: {shadow_free_img_pil.size}, mode: {shadow_free_img_pil.mode}")
            
            # Converter para numpy
            shadow_np = np.array(shadow_img_pil)
            shadow_free_np = np.array(shadow_free_img_pil)
            
            print(f"   ✅ Numpy shadow: {shadow_np.shape}, dtype: {shadow_np.dtype}")
            print(f"   ✅ Numpy shadow-free: {shadow_free_np.shape}, dtype: {shadow_free_np.dtype}")
            
            # Verificar valores
            print(f"   📊 Shadow min/max: {shadow_np.min()}/{shadow_np.max()}")
            print(f"   📊 Shadow-free min/max: {shadow_free_np.min()}/{shadow_free_np.max()}")
            
            # Testar conversão para tensor
            if len(shadow_np.shape) == 3 and shadow_np.shape[2] == 3:
                shadow_tensor = torch.from_numpy(shadow_np).float() / 255.0
                shadow_tensor = shadow_tensor.permute(2, 0, 1)  # HWC -> CHW
                print(f"   ✅ Tensor shadow: {shadow_tensor.shape}, min/max: {shadow_tensor.min():.3f}/{shadow_tensor.max():.3f}")
            else:
                print(f"   ❌ Formato inválido para tensor: {shadow_np.shape}")
                
        except Exception as e:
            print(f"   ❌ Erro PIL: {e}")

if __name__ == "__main__":
    debug_dataset()
