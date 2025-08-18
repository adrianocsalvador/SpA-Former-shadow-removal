#!/usr/bin/env python3
"""
Teste simples do sistema de janela deslizante
"""

import os
import cv2
import numpy as np
import yaml
from attrdict import AttrMap

def test_basic_functionality():
    """Teste básico da funcionalidade de janela deslizante"""
    
    print("=== Teste Básico do Sistema de Janela Deslizante ===")
    
    # Verificar se o dataset existe
    if not os.path.exists('data/ISTD/train/train_A'):
        print("❌ Dataset não encontrado!")
        return False
    
    # Listar imagens disponíveis
    images = [f for f in os.listdir('data/ISTD/train/train_A') if f.endswith('.png')]
    print(f"✅ Encontradas {len(images)} imagens no dataset")
    
    if not images:
        print("❌ Nenhuma imagem encontrada!")
        return False
    
    # Testar carregamento de uma imagem
    test_image = images[0]
    img_path = os.path.join('data/ISTD/train/train_A', test_image)
    
    try:
        img = cv2.imread(img_path, 1)
        if img is None:
            print(f"❌ Não foi possível carregar a imagem: {test_image}")
            return False
        
        h, w = img.shape[:2]
        print(f"✅ Imagem carregada: {test_image} ({w}x{h})")
        
        # Testar criação de janelas
        window_size = (640, 480)
        stride = (320, 240)
        
        windows = []
        for y in range(0, h - window_size[1] + 1, stride[1]):
            for x in range(0, w - window_size[0] + 1, stride[0]):
                windows.append((x, y))
        
        # Adicionar janelas finais se necessário
        if h > window_size[1]:
            y = h - window_size[1]
            for x in range(0, w - window_size[0] + 1, stride[0]):
                windows.append((x, y))
        
        if w > window_size[0]:
            x = w - window_size[0]
            for y in range(0, h - window_size[1] + 1, stride[1]):
                windows.append((x, y))
        
        if h > window_size[1] and w > window_size[0]:
            windows.append((w - window_size[0], h - window_size[1]))
        
        # Remover duplicatas
        windows = list(set(windows))
        
        print(f"✅ Geradas {len(windows)} janelas deslizantes")
        
        # Testar extração de uma janela
        if windows:
            x, y = windows[0]
            window_img = img[y:y+window_size[1], x:x+window_size[0]]
            print(f"✅ Janela extraída: posição ({x},{y}), tamanho {window_img.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro durante o teste: {e}")
        return False

def test_config_loading():
    """Teste de carregamento de configuração"""
    
    print("\n=== Teste de Carregamento de Configuração ===")
    
    config_files = ['config_sliding_window.yml', 'config.yml']
    
    for config_file in config_files:
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r', encoding='UTF-8') as f:
                    config = yaml.load(f, Loader=yaml.FullLoader)
                config = AttrMap(config)
                print(f"✅ Configuração carregada: {config_file}")
                print(f"   - Window size: {getattr(config, 'window_width', 'N/A')}x{getattr(config, 'window_height', 'N/A')}")
                print(f"   - Batch size: {getattr(config, 'batchsize', 'N/A')}")
                return True
            except Exception as e:
                print(f"❌ Erro ao carregar {config_file}: {e}")
    
    print("❌ Nenhuma configuração válida encontrada!")
    return False

def test_dataset_structure():
    """Teste da estrutura do dataset"""
    
    print("\n=== Teste da Estrutura do Dataset ===")
    
    required_dirs = [
        'data/ISTD/train/train_A',
        'data/ISTD/train/train_C',
        'data/ISTD/test/test_A',
        'data/ISTD/test/test_C'
    ]
    
    all_good = True
    
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            files = [f for f in os.listdir(dir_path) if f.endswith('.png')]
            print(f"✅ {dir_path}: {len(files)} imagens")
        else:
            print(f"❌ {dir_path}: não encontrado")
            all_good = False
    
    return all_good

def main():
    """Função principal de teste"""
    
    print("🚀 Iniciando testes do sistema de janela deslizante...\n")
    
    tests = [
        ("Estrutura do Dataset", test_dataset_structure),
        ("Funcionalidade Básica", test_basic_functionality),
        ("Carregamento de Configuração", test_config_loading)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Erro no teste '{test_name}': {e}")
            results.append((test_name, False))
    
    # Resumo dos resultados
    print("\n" + "="*50)
    print("📊 RESUMO DOS TESTES")
    print("="*50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSOU" if result else "❌ FALHOU"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nResultado: {passed}/{total} testes passaram")
    
    if passed == total:
        print("🎉 Todos os testes passaram! O sistema está pronto para uso.")
        print("\nPróximos passos:")
        print("1. python3 demo_sliding_window.py --test_dataset")
        print("2. python3 start_sliding_window_training.py")
    else:
        print("⚠️  Alguns testes falharam. Verifique a configuração.")
    
    return passed == total

if __name__ == '__main__':
    main()
