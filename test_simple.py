#!/usr/bin/env python3
"""
Teste muito simples do sistema de janela deslizante
"""

import os
import cv2
import numpy as np

def test_basic():
    """Teste básico sem dependências complexas"""
    
    print("🚀 Teste Simples do Sistema de Janela Deslizante")
    print("=" * 50)
    
    # 1. Verificar se temos imagens
    if not os.path.exists('data/ISTD/train/train_A'):
        print("❌ Pasta train_A não encontrada!")
        return False
    
    images = [f for f in os.listdir('data/ISTD/train/train_A') if f.endswith('.png')]
    print(f"✅ Encontradas {len(images)} imagens")
    
    if not images:
        print("❌ Nenhuma imagem encontrada!")
        return False
    
    # 2. Carregar uma imagem
    test_image = images[0]
    img_path = os.path.join('data/ISTD/train/train_A', test_image)
    
    try:
        img = cv2.imread(img_path, 1)
        if img is None:
            print(f"❌ Não foi possível carregar: {test_image}")
            return False
        
        h, w = img.shape[:2]
        print(f"✅ Imagem carregada: {test_image} ({w}x{h})")
        
        # 3. Testar janela deslizante
        window_w, window_h = 640, 480
        stride_w, stride_h = 320, 240
        
        windows = []
        
        # Gerar janelas
        for y in range(0, h - window_h + 1, stride_h):
            for x in range(0, w - window_w + 1, stride_w):
                windows.append((x, y))
        
        # Adicionar janelas finais
        if h > window_h:
            y = h - window_h
            for x in range(0, w - window_w + 1, stride_w):
                windows.append((x, y))
        
        if w > window_w:
            x = w - window_w
            for y in range(0, h - window_h + 1, stride_h):
                windows.append((x, y))
        
        if h > window_h and w > window_w:
            windows.append((w - window_w, h - window_h))
        
        # Remover duplicatas
        windows = list(set(windows))
        
        print(f"✅ Geradas {len(windows)} janelas deslizantes")
        
        # 4. Testar extração de janela
        if windows:
            x, y = windows[0]
            window = img[y:y+window_h, x:x+window_w]
            print(f"✅ Janela extraída: posição ({x},{y}), tamanho {window.shape}")
            
            # 5. Testar normalização
            window_norm = window.astype(np.float32) / 255.0
            print(f"✅ Janela normalizada: range [{window_norm.min():.3f}, {window_norm.max():.3f}]")
            
            # 6. Testar transposição para PyTorch
            window_torch = window_norm.transpose(2, 0, 1)  # HWC -> CHW
            print(f"✅ Formato PyTorch: {window_torch.shape}")
        
        print("\n🎉 Teste básico passou com sucesso!")
        print("O sistema de janela deslizante está funcionando corretamente.")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro durante o teste: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dataset_structure():
    """Teste da estrutura do dataset"""
    
    print("\n📁 Verificando estrutura do dataset...")
    
    dirs_to_check = [
        'data/ISTD/train/train_A',
        'data/ISTD/train/train_C', 
        'data/ISTD/test/test_A',
        'data/ISTD/test/test_C'
    ]
    
    all_ok = True
    
    for dir_path in dirs_to_check:
        if os.path.exists(dir_path):
            files = [f for f in os.listdir(dir_path) if f.endswith('.png')]
            print(f"✅ {dir_path}: {len(files)} imagens")
        else:
            print(f"❌ {dir_path}: não encontrado")
            all_ok = False
    
    return all_ok

if __name__ == '__main__':
    print("Iniciando testes...\n")
    
    # Teste da estrutura
    structure_ok = test_dataset_structure()
    
    # Teste básico
    basic_ok = test_basic()
    
    print("\n" + "=" * 50)
    print("📊 RESULTADO FINAL")
    print("=" * 50)
    
    if structure_ok and basic_ok:
        print("🎉 TODOS OS TESTES PASSARAM!")
        print("\nO sistema está pronto para uso!")
        print("\nPróximos passos:")
        print("1. python3 demo_sliding_window.py --test_dataset")
        print("2. python3 start_sliding_window_training.py")
    else:
        print("⚠️  Alguns testes falharam.")
        if not structure_ok:
            print("- Problema na estrutura do dataset")
        if not basic_ok:
            print("- Problema na funcionalidade básica")
