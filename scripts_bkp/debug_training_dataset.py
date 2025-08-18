#!/usr/bin/env python3
"""
Script de debug específico para testar o dataset do script de treinamento
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

# Configurar memória ULTRA
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:128'

def clear_memory():
    """Limpa memória GPU"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    import gc
    gc.collect()

class MemoryOptimizedDataset:
    """Dataset otimizado para memória"""
    
    def __init__(self, data_dir, window_size=512, max_images=50):
        self.data_dir = data_dir
        self.window_size = window_size
        self.max_images = max_images
        self.windows = []
        
        print(f"🔍 Carregando dataset de {data_dir}")
        self._load_windows()
        print(f"✅ Dataset carregado: {len(self.windows)} janelas")
    
    def _load_windows(self):
        """Carrega janelas de forma otimizada"""
        shadow_dir = os.path.join(self.data_dir, 'train_A')
        shadow_free_dir = os.path.join(self.data_dir, 'train_C')
        
        if not os.path.exists(shadow_dir) or not os.path.exists(shadow_free_dir):
            print(f"❌ Diretórios não encontrados: {shadow_dir} ou {shadow_free_dir}")
            return
        
        # Listar imagens e filtrar por tamanho (remover arquivos muito pequenos)
        shadow_files = []
        shadow_free_files = []
        
        for f in os.listdir(shadow_dir):
            if f.endswith(('.jpg', '.jpeg', '.png')):
                file_path = os.path.join(shadow_dir, f)
                if os.path.getsize(file_path) > 10000:  # Filtrar arquivos > 10KB
                    shadow_files.append(f)
        
        for f in os.listdir(shadow_free_dir):
            if f.endswith(('.jpg', '.jpeg', '.png')):
                file_path = os.path.join(shadow_free_dir, f)
                if os.path.getsize(file_path) > 10000:  # Filtrar arquivos > 10KB
                    shadow_free_files.append(f)
        
        # Limitar número de imagens
        shadow_files = shadow_files[:self.max_images]
        shadow_free_files = shadow_free_files[:self.max_images]
        
        print(f"📁 Encontradas {len(shadow_files)} imagens válidas com sombra e {len(shadow_free_files)} sem sombra")
        
        # Processar cada par de imagens
        for shadow_file in shadow_files:
            # Usar o mesmo nome de arquivo para ambos os diretórios
            shadow_free_file = shadow_file
            if shadow_free_file not in shadow_free_files:
                continue
                
            shadow_path = os.path.join(shadow_dir, shadow_file)
            shadow_free_path = os.path.join(shadow_free_dir, shadow_free_file)
            
            try:
                # Carregar imagens
                shadow_img = cv2.imread(shadow_path)
                shadow_free_img = cv2.imread(shadow_free_path)
                
                # Debug: verificar se as imagens foram carregadas
                if shadow_img is None:
                    print(f"❌ Falha ao carregar imagem: {shadow_path}")
                    continue
                if shadow_free_img is None:
                    print(f"❌ Falha ao carregar imagem: {shadow_free_path}")
                    continue
                
                # Verificar dimensões
                if shadow_img.shape != shadow_free_img.shape:
                    print(f"⚠️ Dimensões diferentes: {shadow_file} - {shadow_img.shape} vs {shadow_free_img.shape}")
                    continue
                
                # Verificar se tem 3 canais
                if len(shadow_img.shape) != 3 or shadow_img.shape[2] != 3:
                    print(f"⚠️ Imagem sem 3 canais: {shadow_file} - {shadow_img.shape}")
                    continue
                
                # Converter BGR para RGB
                shadow_img = cv2.cvtColor(shadow_img, cv2.COLOR_BGR2RGB)
                shadow_free_img = cv2.cvtColor(shadow_free_img, cv2.COLOR_BGR2RGB)
                
                # Verificar se a imagem tem o tamanho esperado
                h, w = shadow_img.shape[:2]
                if h == self.window_size and w == self.window_size:
                    # A imagem já é do tamanho da janela
                    self.windows.append({
                        'shadow': shadow_img.copy(),
                        'shadow_free': shadow_free_img.copy(),
                        'x': 0, 'y': 0
                    })
                    print(f"✅ Adicionada janela: {shadow_file} - {shadow_img.shape}")
                else:
                    # Extrair janelas se a imagem for maior
                    for y in range(0, h - self.window_size + 1, self.window_size // 2):
                        for x in range(0, w - self.window_size + 1, self.window_size // 2):
                            shadow_window = shadow_img[y:y+self.window_size, x:x+self.window_size]
                            shadow_free_window = shadow_free_img[y:y+self.window_size, x:x+self.window_size]
                            
                            # Verificar se a janela tem conteúdo válido
                            if shadow_window.shape == (self.window_size, self.window_size, 3):
                                self.windows.append({
                                    'shadow': shadow_window.copy(),
                                    'shadow_free': shadow_free_window.copy(),
                                    'x': x, 'y': y
                                })
                
                # Limpar memória após cada imagem
                del shadow_img, shadow_free_img
                clear_memory()
                
            except Exception as e:
                print(f"❌ Erro ao processar {shadow_file}: {e}")
                continue
    
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        window = self.windows[idx]
        
        # Verificar se os dados são válidos
        if window['shadow'] is None or window['shadow_free'] is None:
            print(f"⚠️ Dados inválidos no índice {idx}")
            # Retornar dados dummy
            dummy_data = np.zeros((self.window_size, self.window_size, 3), dtype=np.uint8)
            shadow = dummy_data.astype(np.float32) / 255.0
            shadow_free = dummy_data.astype(np.float32) / 255.0
        else:
            # Normalizar para [0, 1]
            shadow = window['shadow'].astype(np.float32) / 255.0
            shadow_free = window['shadow_free'].astype(np.float32) / 255.0
        
        # Verificar dimensões
        if shadow.shape != (self.window_size, self.window_size, 3):
            print(f"⚠️ Dimensões incorretas no índice {idx}: {shadow.shape}")
            # Retornar dados dummy
            dummy_data = np.zeros((self.window_size, self.window_size, 3), dtype=np.uint8)
            shadow = dummy_data.astype(np.float32) / 255.0
            shadow_free = dummy_data.astype(np.float32) / 255.0
        
        # Converter para tensor
        shadow = torch.from_numpy(shadow).permute(2, 0, 1)  # HWC -> CHW
        shadow_free = torch.from_numpy(shadow_free).permute(2, 0, 1)
        
        # Verificar se os tensores têm valores válidos
        if torch.isnan(shadow).any() or torch.isinf(shadow).any():
            print(f"⚠️ Valores NaN/Inf no índice {idx}")
            shadow = torch.zeros(3, self.window_size, self.window_size)
            shadow_free = torch.zeros(3, self.window_size, self.window_size)
        
        return shadow, shadow_free

def debug_training_dataset():
    """Debug do dataset de treinamento"""
    data_dir = "/mnt/48EC7EE9EC7ED0A4/Teste_Sombra/dataset_v9_pairs_512x512/train"
    
    print("🧪 Testando dataset de treinamento...")
    
    # Criar dataset
    dataset = MemoryOptimizedDataset(data_dir, window_size=512, max_images=5)
    
    print(f"📊 Dataset criado com {len(dataset)} janelas")
    
    # Testar alguns itens
    print("\n🧪 Testando itens do dataset...")
    
    for i in range(min(3, len(dataset))):
        print(f"\n📸 Testando item {i}:")
        
        try:
            shadow, shadow_free = dataset[i]
            
            print(f"   ✅ Shadow tensor: {shadow.shape}, dtype: {shadow.dtype}")
            print(f"   ✅ Shadow-free tensor: {shadow_free.shape}, dtype: {shadow_free.dtype}")
            print(f"   📊 Shadow min/max: {shadow.min():.3f}/{shadow.max():.3f}")
            print(f"   📊 Shadow-free min/max: {shadow_free.min():.3f}/{shadow_free.max():.3f}")
            
            # Verificar se tem valores válidos
            if torch.isnan(shadow).any():
                print(f"   ❌ Shadow tem valores NaN!")
            if torch.isinf(shadow).any():
                print(f"   ❌ Shadow tem valores Inf!")
            if torch.isnan(shadow_free).any():
                print(f"   ❌ Shadow-free tem valores NaN!")
            if torch.isinf(shadow_free).any():
                print(f"   ❌ Shadow-free tem valores Inf!")
            
            # Verificar se tem 3 canais
            if shadow.shape[0] != 3:
                print(f"   ❌ Shadow não tem 3 canais: {shadow.shape}")
            if shadow_free.shape[0] != 3:
                print(f"   ❌ Shadow-free não tem 3 canais: {shadow_free.shape}")
                
        except Exception as e:
            print(f"   ❌ Erro ao carregar item {i}: {e}")
    
    # Testar DataLoader
    print("\n🧪 Testando DataLoader...")
    
    try:
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
        
        for batch_idx, (shadow_batch, shadow_free_batch) in enumerate(dataloader):
            print(f"\n📦 Batch {batch_idx}:")
            print(f"   ✅ Shadow batch: {shadow_batch.shape}, dtype: {shadow_batch.dtype}")
            print(f"   ✅ Shadow-free batch: {shadow_free_batch.shape}, dtype: {shadow_free_batch.dtype}")
            print(f"   📊 Shadow batch min/max: {shadow_batch.min():.3f}/{shadow_batch.max():.3f}")
            print(f"   📊 Shadow-free batch min/max: {shadow_free_batch.min():.3f}/{shadow_free_batch.max():.3f}")
            
            # Verificar se tem valores válidos
            if torch.isnan(shadow_batch).any():
                print(f"   ❌ Shadow batch tem valores NaN!")
            if torch.isinf(shadow_batch).any():
                print(f"   ❌ Shadow batch tem valores Inf!")
            
            # Testar apenas o primeiro batch
            if batch_idx == 0:
                break
                
    except Exception as e:
        print(f"❌ Erro no DataLoader: {e}")

if __name__ == "__main__":
    debug_training_dataset()
