#!/usr/bin/env python3
"""
Script para comparar diferentes abordagens para processamento de imagens panorâmicas
"""

import torch
import torch.nn as nn
import numpy as np
import time
import psutil
import os
from SpA_Former import Generator
from SpA_Former_Light import Generator_Light


def get_gpu_memory_usage():
    """Obtém uso atual de memória da GPU"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3  # GB
    return 0


def get_system_memory_usage():
    """Obtém uso atual de memória do sistema"""
    return psutil.virtual_memory().used / 1024**3  # GB


def estimate_memory_requirements(image_size, model_type="original"):
    """Estima requisitos de memória para diferentes abordagens"""
    
    # Tamanho da imagem em pixels
    w, h = image_size
    pixels = w * h
    
    # Memória por pixel (float32)
    bytes_per_pixel = 4
    
    # Estimativas de memória
    estimates = {}
    
    # 1. Abordagem Original (sem otimizações)
    estimates["original"] = {
        "input_memory": pixels * 3 * bytes_per_pixel / 1024**3,  # GB
        "model_memory": 2.5,  # GB (estimativa)
        "gradients_memory": 1.5,  # GB
        "total_estimated": pixels * 3 * bytes_per_pixel / 1024**3 + 4.0
    }
    
    # 2. Sliding Window Otimizado
    window_size = 1024 * 1024
    num_windows = (w // 512 + 1) * (h // 512 + 1)
    estimates["sliding_window"] = {
        "input_memory": window_size * 3 * bytes_per_pixel / 1024**3,  # GB
        "model_memory": 2.5,  # GB
        "gradients_memory": 1.5,  # GB
        "total_estimated": window_size * 3 * bytes_per_pixel / 1024**3 + 4.0,
        "num_windows": num_windows
    }
    
    # 3. Modelo Leve
    estimates["light_model"] = {
        "input_memory": pixels * 3 * bytes_per_pixel / 1024**3,  # GB
        "model_memory": 1.2,  # GB (reduzido)
        "gradients_memory": 0.8,  # GB (reduzido)
        "total_estimated": pixels * 3 * bytes_per_pixel / 1024**3 + 2.0
    }
    
    # 4. Mixed Precision
    estimates["mixed_precision"] = {
        "input_memory": pixels * 3 * bytes_per_pixel / 1024**3 / 2,  # GB (FP16)
        "model_memory": 2.5 / 2,  # GB (FP16)
        "gradients_memory": 1.5 / 2,  # GB (FP16)
        "total_estimated": (pixels * 3 * bytes_per_pixel / 1024**3 + 4.0) / 2
    }
    
    return estimates


def test_model_memory_usage(model, input_size, model_name):
    """Testa uso de memória de um modelo específico"""
    print(f"\n=== Testando {model_name} ===")
    
    # Criar tensor de entrada
    input_tensor = torch.randn(1, 3, input_size[1], input_size[0])
    
    if torch.cuda.is_available():
        input_tensor = input_tensor.cuda()
        model = model.cuda()
        
        # Limpar cache
        torch.cuda.empty_cache()
        
        # Medir memória antes
        memory_before = get_gpu_memory_usage()
        
        try:
            # Forward pass
            with torch.no_grad():
                output = model(input_tensor)
            
            # Medir memória após forward
            memory_after_forward = get_gpu_memory_usage()
            
            # Backward pass (se necessário)
            if hasattr(output, 'backward'):
                output.backward(torch.ones_like(output))
            
            # Medir memória após backward
            memory_after_backward = get_gpu_memory_usage()
            
            print(f"Memória antes: {memory_before:.2f} GB")
            print(f"Memória após forward: {memory_after_forward:.2f} GB")
            print(f"Memória após backward: {memory_after_backward:.2f} GB")
            print(f"Incremento forward: {memory_after_forward - memory_before:.2f} GB")
            print(f"Incremento backward: {memory_after_backward - memory_after_forward:.2f} GB")
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"❌ ERRO: Memória insuficiente para {model_name}")
                print(f"Erro: {e}")
            else:
                print(f"❌ Erro ao testar {model_name}: {e}")
        
        finally:
            # Limpar cache
            torch.cuda.empty_cache()
    
    else:
        print("CUDA não disponível")


def compare_approaches():
    """Compara diferentes abordagens"""
    
    print("=== COMPARAÇÃO DE ABORDAGENS PARA IMAGENS PANORÂMICAS ===\n")
    
    # Tamanho da imagem panorâmica
    image_size = (12288, 6144)
    print(f"Tamanho da imagem: {image_size[0]}x{image_size[1]} pixels")
    print(f"Total de pixels: {image_size[0] * image_size[1]:,}")
    
    # Estimativas de memória
    estimates = estimate_memory_requirements(image_size)
    
    print("\n=== ESTIMATIVAS DE MEMÓRIA ===")
    for approach, estimate in estimates.items():
        print(f"\n{approach.upper()}:")
        print(f"  Input: {estimate['input_memory']:.2f} GB")
        print(f"  Modelo: {estimate['model_memory']:.2f} GB")
        print(f"  Gradientes: {estimate['gradients_memory']:.2f} GB")
        print(f"  Total estimado: {estimate['total_estimated']:.2f} GB")
        if 'num_windows' in estimate:
            print(f"  Número de janelas: {estimate['num_windows']}")
    
    # Testar modelos reais
    print("\n=== TESTES PRÁTICOS ===")
    
    # Testar com janela menor primeiro
    test_sizes = [
        (1024, 1024),    # Janela padrão
        (2048, 2048),    # Janela maior
        (4096, 4096),    # Janela muito grande
    ]
    
    for test_size in test_sizes:
        print(f"\n--- Testando com tamanho {test_size[0]}x{test_size[1]} ---")
        
        # Modelo original
        try:
            model_original = Generator(gpu_ids=[0] if torch.cuda.is_available() else [])
            test_model_memory_usage(model_original, test_size, "Modelo Original")
        except Exception as e:
            print(f"Erro ao criar modelo original: {e}")
        
        # Modelo leve
        try:
            model_light = Generator_Light(gpu_ids=[0] if torch.cuda.is_available() else [])
            test_model_memory_usage(model_light, test_size, "Modelo Leve")
        except Exception as e:
            print(f"Erro ao criar modelo leve: {e}")
    
    # Recomendações
    print("\n=== RECOMENDAÇÕES ===")
    print("1. SLIDING WINDOW (Recomendado):")
    print("   - Use janelas de 1024x1024 com stride de 512")
    print("   - Implemente blending para suavizar transições")
    print("   - Use mixed precision (FP16) para economizar memória")
    
    print("\n2. OTIMIZAÇÕES DE MEMÓRIA:")
    print("   - Gradient checkpointing")
    print("   - Gradient accumulation")
    print("   - Batch size = 1")
    print("   - Mixed precision training")
    
    print("\n3. MODELO LEVE (Alternativa):")
    print("   - Reduz parâmetros em ~50%")
    print("   - Mantém qualidade aceitável")
    print("   - Pode processar janelas maiores")
    
    print("\n4. CONFIGURAÇÃO SUGERIDA PARA RTX 4060:")
    print("   - Window size: 1024x1024")
    print("   - Stride: 512x512")
    print("   - Batch size: 1")
    print("   - Mixed precision: True")
    print("   - Gradient accumulation: 4 steps")


if __name__ == "__main__":
    compare_approaches()
