#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import argparse
import cv2
import matplotlib.pyplot as plt
import time
import torch
from torch.autograd import Variable
from utils import gpu_manage, heatmap
from SpA_Former import Generator

def test_model_epoch_34(test_image_path, model_path, output_path=None):
    """
    Testa o modelo da época 34 com uma imagem específica
    """
    print(f"🔍 Testando modelo da época 34...")
    print(f"📸 Imagem de teste: {test_image_path}")
    print(f"🤖 Modelo: {model_path}")
    
    # Configurações
    width = 320
    height = 240
    
    # Verificar se o modelo existe
    if not os.path.exists(model_path):
        print(f"❌ Erro: Modelo não encontrado: {model_path}")
        return False
    
    # Carregar imagem de teste
    if not os.path.exists(test_image_path):
        print(f"❌ Erro: Imagem de teste não encontrada: {test_image_path}")
        return False
    
    # Carregar e redimensionar imagem
    img = cv2.imread(test_image_path)
    if img is None:
        print(f"❌ Erro: Não foi possível carregar a imagem: {test_image_path}")
        return False
    
    # Redimensionar para 320x240 (tamanho do treinamento)
    img_resized = cv2.resize(img, (width, height))
    
    # Converter BGR para RGB
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    
    # Normalizar para [0, 1]
    img_normalized = img_rgb.astype(np.float32) / 255.0
    
    # Converter para tensor PyTorch
    img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).unsqueeze(0)
    
    # Configurar GPU/CPU
    class SimpleConfig:
        def __init__(self):
            self.cuda = torch.cuda.is_available()
            self.gpu_ids = [0] if self.cuda else []
            self.manualSeed = 42
    
    config = SimpleConfig()
    gpu_manage(config)
    device = torch.device('cuda' if config.cuda else 'cpu')
    img_tensor = img_tensor.to(device)
    
    # Carregar modelo
    print("🔄 Carregando modelo da época 34...")
    try:
        generator = Generator(gpu_ids=config.gpu_ids)
        generator.load_state_dict(torch.load(model_path, map_location=device))
        generator.to(device)
        generator.eval()
        print("✅ Modelo carregado com sucesso!")
    except Exception as e:
        print(f"❌ Erro ao carregar modelo: {e}")
        return False
    
    # Fazer predição
    print("🚀 Fazendo predição...")
    with torch.no_grad():
        start_time = time.time()
        result = generator(img_tensor)
        inference_time = time.time() - start_time
    
    # O modelo retorna (attention, output)
    if isinstance(result, tuple):
        attention, output = result
        print(f"📊 Attention shape: {attention.shape}")
    else:
        output = result
    
    # Converter saída para numpy
    output_np = output.squeeze().cpu().numpy()
    output_np = np.transpose(output_np, (1, 2, 0))
    
    # Clamp para [0, 1] e converter para uint8
    output_np = np.clip(output_np, 0, 1)
    output_img = (output_np * 255).astype(np.uint8)
    
    # Converter de volta para BGR para salvar
    output_bgr = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
    
    # Salvar resultado
    if output_path is None:
        output_path = f"test_epoch_34_result_{int(time.time())}.png"
    
    cv2.imwrite(output_path, output_bgr)
    print(f"💾 Resultado salvo: {output_path}")
    print(f"⏱️ Tempo de inferência: {inference_time:.3f}s")
    
    # Mostrar comparação
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Imagem original
    axes[0].imshow(img_rgb)
    axes[0].set_title('Imagem Original (com sombra)')
    axes[0].axis('off')
    
    # Imagem redimensionada
    axes[1].imshow(img_rgb)
    axes[1].set_title(f'Imagem Redimensionada ({width}x{height})')
    axes[1].axis('off')
    
    # Resultado
    axes[2].imshow(output_img)
    axes[2].set_title('Resultado (sem sombra) - Época 34')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    # Salvar comparação
    comparison_path = f"test_epoch_34_comparison_{int(time.time())}.png"
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 Comparação salva: {comparison_path}")
    
    print("✅ Teste concluído com sucesso!")
    return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Testar modelo da época 34')
    parser.add_argument('--test_image', type=str, 
                       default='/home/consultoria/Desktop/shadow_test1/20250715_01_00017_PAN.jpeg',
                       help='Caminho para imagem de teste')
    parser.add_argument('--model_path', type=str,
                       default='results_panoramic/models/gen_model_epoch_34.pth',
                       help='Caminho para o modelo da época 34')
    parser.add_argument('--output', type=str, default=None,
                       help='Caminho para salvar o resultado')
    
    args = parser.parse_args()
    
    success = test_model_epoch_34(args.test_image, args.model_path, args.output)
    
    if success:
        print("\n🎉 Teste do modelo da época 34 realizado com sucesso!")
        print("📁 Arquivos gerados:")
        print(f"   - Resultado: {args.output or 'test_epoch_34_result_*.png'}")
        print(f"   - Comparação: test_epoch_34_comparison_*.png")
    else:
        print("\n❌ Falha no teste do modelo da época 34") 