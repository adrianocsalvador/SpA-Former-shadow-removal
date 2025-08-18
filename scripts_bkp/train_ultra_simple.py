#!/usr/bin/env python3
"""
Treinamento ULTRA simples para memória com janelas 512x512
"""

import os
import yaml
import time
import torch
import datetime
import gc

# Configurar memória ULTRA
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:128'

def clear_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    gc.collect()

def train_ultra_simple():
    # Setup inicial
    clear_memory()
    
    # Carregar configuração
    with open('config_training_v9_pairs_512x512_ultra_memory.yml', 'r') as f:
        config_dict = yaml.safe_load(f)
    
    class SimpleConfig:
        def __init__(self, config_dict):
            for key, value in config_dict.items():
                setattr(self, key, value)
    
    config = SimpleConfig(config_dict)
    
    # Criar diretório com timestamp
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M')
    output_dir = f"./results_{timestamp}_ultra"
    config.out_dir = output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    print(f'📁 Resultados salvos em: {output_dir}')
    print('🚀 Iniciando treinamento ULTRA otimizado...')
    
    # Log de início
    with open(os.path.join(output_dir, 'training_log.txt'), 'w') as f:
        f.write(f"=== TREINAMENTO ULTRA SIMPLES ===\n")
        f.write(f"Início: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Configuração: {config.dataset_version}\n")
        f.write("=" * 50 + "\n\n")
    
    # Simular treinamento (você pode substituir pelo treinamento real)
    for epoch in range(1, config.epoch + 1):
        epoch_start = time.time()
        
        # Simular processamento
        print(f"Época {epoch}/{config.epoch} - Processando...")
        
        # Limpeza de memória a cada época
        clear_memory()
        
        epoch_time = time.time() - epoch_start
        
        # Log
        with open(os.path.join(output_dir, 'training_log.txt'), 'a') as f:
            f.write(f"Época {epoch:3d}/{config.epoch:3d} | Tempo: {epoch_time:.2f}s\n")
        
        print(f"  ✅ Época {epoch} concluída em {epoch_time:.2f}s")
    
    total_time = time.time() - epoch_start
    print(f'🎉 Treinamento concluído em {total_time:.2f}s')
    print(f'📁 Resultados salvos em: {output_dir}')

if __name__ == '__main__':
    train_ultra_simple()


