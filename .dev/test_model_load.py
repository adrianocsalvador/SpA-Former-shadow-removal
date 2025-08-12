#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import yaml
from SpA_Former import Generator

def test_model_load():
    """Testa se o modelo SpA-Former carrega sem problemas"""
    
    print("🧪 Testando carregamento do modelo SpA-Former...")
    
    # Configuração mínima
    config = {
        'gpu_ids': [0],
        'width': 320,
        'height': 240,
        'in_ch': 3,
        'out_ch': 3
    }
    
    try:
        # Criar modelo
        print("📦 Criando modelo Generator...")
        gen = Generator(gpu_ids=config['gpu_ids'])
        
        # Mover para GPU
        print("🚀 Movendo modelo para GPU...")
        gen = gen.cuda()
        
        # Criar tensor de teste pequeno
        print("📊 Criando tensor de teste...")
        test_input = torch.randn(1, 3, 320, 240).cuda()
        
        # Forward pass
        print("🔄 Executando forward pass...")
        with torch.no_grad():
            att, fake_b = gen.forward(test_input)
        
        print(f"✅ Modelo carregado com sucesso!")
        print(f"📊 Input shape: {test_input.shape}")
        print(f"📊 Output shape: {fake_b.shape}")
        print(f"📊 Attention shape: {att.shape if att is not None else 'None'}")
        
        # Limpar memória
        del gen, test_input, att, fake_b
        torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        print(f"❌ Erro ao carregar modelo: {e}")
        return False

if __name__ == '__main__':
    test_model_load() 