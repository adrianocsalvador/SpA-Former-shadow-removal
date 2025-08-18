#!/usr/bin/env python3
"""
Script de teste rápido para verificar configuração panorâmica
"""

import torch
import yaml
import os
import cv2
import numpy as np
from SpA_Former import Generator
from SpA_Former_Light import Generator_Light


def test_gpu_memory():
    """Testa memória disponível na GPU"""
    print("=== TESTE DE MEMÓRIA GPU ===")
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        allocated_memory = torch.cuda.memory_allocated(0) / 1024**3
        free_memory = total_memory - allocated_memory
        
        print(f"GPU: {gpu_name}")
        print(f"Memória total: {total_memory:.2f} GB")
        print(f"Memória alocada: {allocated_memory:.2f} GB")
        print(f"Memória livre: {free_memory:.2f} GB")
        
        return free_memory
    else:
        print("❌ CUDA não disponível")
        return 0


def test_model_loading():
    """Testa carregamento dos modelos"""
    print("\n=== TESTE DE CARREGAMENTO DE MODELOS ===")
    
    try:
        # Testar modelo original
        print("Carregando modelo original...")
        model_original = Generator(gpu_ids=[0] if torch.cuda.is_available() else [])
        print("✅ Modelo original carregado com sucesso")
        
        # Contar parâmetros
        total_params = sum(p.numel() for p in model_original.parameters())
        print(f"Parâmetros do modelo original: {total_params:,}")
        
        # Testar modelo leve
        print("Carregando modelo leve...")
        model_light = Generator_Light(gpu_ids=[0] if torch.cuda.is_available() else [])
        print("✅ Modelo leve carregado com sucesso")
        
        total_params_light = sum(p.numel() for p in model_light.parameters())
        print(f"Parâmetros do modelo leve: {total_params_light:,}")
        print(f"Redução: {((total_params - total_params_light) / total_params * 100):.1f}%")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro ao carregar modelos: {e}")
        return False


def test_memory_usage_with_different_sizes():
    """Testa uso de memória com diferentes tamanhos de entrada"""
    print("\n=== TESTE DE USO DE MEMÓRIA ===")
    
    if not torch.cuda.is_available():
        print("❌ CUDA não disponível para teste de memória")
        return
    
    # Tamanhos de teste
    test_sizes = [
        (256, 256),    # Muito pequeno
        (512, 512),    # Pequeno
        (1024, 1024),  # Médio
        (2048, 2048),  # Grande
    ]
    
    model = Generator(gpu_ids=[0])
    model.cuda()
    
    for size in test_sizes:
        print(f"\nTestando tamanho {size[0]}x{size[1]}:")
        
        # Limpar cache
        torch.cuda.empty_cache()
        
        try:
            # Criar tensor de entrada
            input_tensor = torch.randn(1, 3, size[1], size[0]).cuda()
            
            # Medir memória antes
            memory_before = torch.cuda.memory_allocated(0) / 1024**3
            
            # Forward pass
            with torch.no_grad():
                output = model(input_tensor)
            
            # Medir memória após
            memory_after = torch.cuda.memory_allocated(0) / 1024**3
            memory_used = memory_after - memory_before
            
            print(f"  Memória usada: {memory_used:.2f} GB")
            print(f"  ✅ Sucesso")
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"  ❌ Memória insuficiente")
                break
            else:
                print(f"  ❌ Erro: {e}")
                break


def test_config_file():
    """Testa arquivo de configuração"""
    print("\n=== TESTE DE CONFIGURAÇÃO ===")
    
    config_file = "config_panoramic.yml"
    
    if not os.path.exists(config_file):
        print(f"❌ Arquivo de configuração não encontrado: {config_file}")
        return False
    
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        print("✅ Arquivo de configuração carregado com sucesso")
        
        # Verificar parâmetros importantes
        important_params = [
            'window_width', 'window_height', 'stride_width', 'stride_height',
            'batchsize', 'use_mixed_precision', 'gradient_accumulation_steps'
        ]
        
        for param in important_params:
            if param in config:
                print(f"  {param}: {config[param]}")
            else:
                print(f"  ❌ {param}: não encontrado")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro ao carregar configuração: {e}")
        return False


def test_dataset_structure():
    """Testa estrutura do dataset"""
    print("\n=== TESTE DE ESTRUTURA DO DATASET ===")
    
    # Verificar se existe estrutura básica
    required_dirs = [
        "./data/panoramic/train/train_A",
        "./data/panoramic/train/train_C",
        "./data/panoramic/test/test_A",
        "./data/panoramic/test/test_C"
    ]
    
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            files = len([f for f in os.listdir(dir_path) if f.endswith(('.png', '.jpg', '.jpeg'))])
            print(f"✅ {dir_path}: {files} imagens")
        else:
            print(f"❌ {dir_path}: não encontrado")
    
    # Criar estrutura se não existir
    print("\nCriando estrutura de diretórios...")
    for dir_path in required_dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"  Criado: {dir_path}")


def create_test_image():
    """Cria uma imagem de teste"""
    print("\n=== CRIANDO IMAGEM DE TESTE ===")
    
    # Criar imagem de teste simples
    test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    
    # Adicionar "sombra" artificial
    test_image[200:300, 200:300] = test_image[200:300, 200:300] // 2
    
    # Salvar imagem de teste
    test_dir = "./data/panoramic/test/test_A"
    os.makedirs(test_dir, exist_ok=True)
    
    test_path = os.path.join(test_dir, "test_image.png")
    cv2.imwrite(test_path, test_image)
    
    # Criar versão sem sombra (target)
    target_dir = "./data/panoramic/test/test_C"
    os.makedirs(target_dir, exist_ok=True)
    
    target_path = os.path.join(target_dir, "test_image.png")
    cv2.imwrite(target_path, test_image)
    
    print(f"✅ Imagem de teste criada: {test_path}")
    return test_path


def main():
    """Função principal de teste"""
    print("🚀 INICIANDO TESTES DA CONFIGURAÇÃO PANORÂMICA")
    print("=" * 50)
    
    # Teste 1: Memória GPU
    free_memory = test_gpu_memory()
    
    # Teste 2: Configuração
    config_ok = test_config_file()
    
    # Teste 3: Modelos
    models_ok = test_model_loading()
    
    # Teste 4: Uso de memória
    test_memory_usage_with_different_sizes()
    
    # Teste 5: Estrutura do dataset
    test_dataset_structure()
    
    # Teste 6: Criar imagem de teste
    test_image_path = create_test_image()
    
    # Resumo
    print("\n" + "=" * 50)
    print("📋 RESUMO DOS TESTES")
    print("=" * 50)
    
    if free_memory > 2.0:
        print("✅ Memória GPU: Suficiente")
    else:
        print("⚠️ Memória GPU: Pode ser insuficiente")
    
    if config_ok:
        print("✅ Configuração: OK")
    else:
        print("❌ Configuração: Problemas encontrados")
    
    if models_ok:
        print("✅ Modelos: Carregados com sucesso")
    else:
        print("❌ Modelos: Problemas no carregamento")
    
    print(f"✅ Imagem de teste: {test_image_path}")
    
    print("\n🎯 PRÓXIMOS PASSOS:")
    print("1. Execute: python compare_approaches.py")
    print("2. Execute: python predict_panoramic.py --test_dir ./data/panoramic/test/test_A --out_dir ./test_output --pretrained [SEU_MODELO] --cuda --window_size 512 --stride 256")
    print("3. Se tudo funcionar, aumente os parâmetros gradualmente")


if __name__ == "__main__":
    main()
