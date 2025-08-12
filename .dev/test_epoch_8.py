#!/usr/bin/env python3

import os
import sys
import torch
import cv2
import numpy as np
import argparse
from PIL import Image
import matplotlib.pyplot as plt

# Adicionar o diretório raiz ao path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from SpA_Former import Generator
import utils

class SimpleConfig:
    """Classe simples para configuração"""
    def __init__(self):
        self.cuda = True
        self.gpu_ids = [0]
        self.width = 320
        self.height = 240

def extract_epoch_number(model_path):
    """Extrai o número da época do caminho do modelo"""
    import re
    match = re.search(r'epoch_(\d+)', model_path)
    if match:
        return match.group(1)
    return "unknown"

def main():
    parser = argparse.ArgumentParser(description='Teste do Modelo SpA-Former')
    parser.add_argument('--model', type=str, required=True,
                       help='Caminho para o modelo treinado')
    parser.add_argument('--input', type=str, required=True,
                       help='Caminho para a imagem de entrada')
    parser.add_argument('--output_dir', type=str, default='./out',
                       help='Diretório de saída (padrão: ./out)')
    
    args = parser.parse_args()
    
    # Extrair número da época do nome do modelo
    epoch_number = extract_epoch_number(args.model)
    
    print("🧪 Teste do Modelo SpA-Former")
    print("=" * 50)
    print(f"🧪 Testando modelo: {args.model}")
    print(f"📸 Imagem de entrada: {args.input}")
    print(f"📊 Época do modelo: {epoch_number}")
    
    # Criar diretório de saída
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Verificar se GPU está disponível
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        print(f"🚀 Usando GPU: {torch.cuda.get_device_name()}")
    else:
        print("💻 Usando CPU")
    
    # Carregar modelo
    print("📦 Carregando modelo...")
    gen = Generator(gpu_ids=[0] if torch.cuda.is_available() else [])
    gen.load_state_dict(torch.load(args.model, map_location='cpu'))
    gen.to(device)
    gen.eval()
    print("✅ Modelo carregado com sucesso!")
    
    # Carregar e processar imagem
    print("🖼️ Processando imagem de entrada...")
    input_img = cv2.imread(args.input)
    if input_img is None:
        print(f"❌ Erro ao carregar imagem: {args.input}")
        return
    
    original_height, original_width = input_img.shape[:2]
    print(f"📊 Dimensões originais: {input_img.shape}")
    
    # Redimensionar para processamento (320x240)
    input_resized = cv2.resize(input_img, (320, 240), interpolation=cv2.INTER_LANCZOS4)
    print(f"📊 Dimensões para processamento: {input_resized.shape}")
    
    # Converter para tensor
    input_tensor = torch.from_numpy(input_resized).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    input_tensor = input_tensor.to(device)
    
    # Fazer predição
    print("🔮 Fazendo predição...")
    with torch.no_grad():
        attention, output = gen(input_tensor)
    
    # Converter saída para numpy
    output_low_res = output.squeeze().permute(1, 2, 0).cpu().numpy()
    output_low_res = (output_low_res * 255).astype(np.uint8)
    
    # Redimensionar a saída para a resolução original
    output_high_res = cv2.resize(output_low_res, (original_width, original_height), interpolation=cv2.INTER_LANCZOS4)
    print(f"📊 Dimensões da saída final: {output_high_res.shape}")
    
    # Aplicar filtro de suavização para reduzir pixelização
    output_high_res_smooth = cv2.GaussianBlur(output_high_res, (3, 3), 0.5)
    # Misturar imagem original com suavizada para preservar detalhes
    output_high_res_final = cv2.addWeighted(output_high_res, 0.7, output_high_res_smooth, 0.3, 0)
    
    # Alternativa: Upscaling progressivo para melhor qualidade
    def progressive_upscale(image, target_width, target_height):
        """Upscaling progressivo para melhor qualidade"""
        current_width, current_height = image.shape[1], image.shape[0]
        
        # Primeira etapa: redimensionar para um tamanho intermediário
        intermediate_width = min(target_width // 2, current_width * 2)
        intermediate_height = min(target_height // 2, current_height * 2)
        
        intermediate = cv2.resize(image, (intermediate_width, intermediate_height), interpolation=cv2.INTER_CUBIC)
        
        # Segunda etapa: redimensionar para o tamanho final
        final = cv2.resize(intermediate, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)
        
        return final
    
    output_high_res_progressive = progressive_upscale(output_low_res, original_width, original_height)
    
    # Nome base do arquivo
    input_filename = os.path.splitext(os.path.basename(args.input))[0]
    
    # Salvar imagem de entrada original
    input_original_save_path = os.path.join(args.output_dir, f"{input_filename}_input_original_epoch{epoch_number}.png")
    cv2.imwrite(input_original_save_path, input_img)
    print(f"💾 Imagem de entrada original salva: {input_original_save_path}")
    
    # Salvar imagem de entrada redimensionada
    input_resized_save_path = os.path.join(args.output_dir, f"{input_filename}_input_resized_epoch{epoch_number}.png")
    cv2.imwrite(input_resized_save_path, input_resized)
    print(f"💾 Imagem de entrada redimensionada salva: {input_resized_save_path}")
    
    # Salvar imagem de saída em alta resolução
    output_save_path = os.path.join(args.output_dir, f"{input_filename}_output_highres_epoch{epoch_number}.png")
    cv2.imwrite(output_save_path, output_high_res_final)
    print(f"💾 Imagem de saída em alta resolução salva: {output_save_path}")
    
    # Salvar versão com upscaling progressivo (sem suavização)
    output_progressive_save_path = os.path.join(args.output_dir, f"{input_filename}_output_progressive_epoch{epoch_number}.png")
    cv2.imwrite(output_progressive_save_path, output_high_res_progressive)
    print(f"💾 Imagem com upscaling progressivo salva: {output_progressive_save_path}")
    
    # Salvar versão original (sem melhorias)
    output_original_save_path = os.path.join(args.output_dir, f"{input_filename}_output_original_epoch{epoch_number}.png")
    cv2.imwrite(output_original_save_path, output_high_res)
    print(f"💾 Imagem original (sem melhorias) salva: {output_original_save_path}")
    
    # Salvar imagem de saída em baixa resolução
    output_low_res_save_path = os.path.join(args.output_dir, f"{input_filename}_output_lowres_epoch{epoch_number}.png")
    cv2.imwrite(output_low_res_save_path, output_low_res)
    print(f"💾 Imagem de saída em baixa resolução salva: {output_low_res_save_path}")
    
    # Criar comparação lado a lado (alta resolução)
    comparison_high_res = np.hstack([input_img, output_high_res_final])
    comparison_high_res_save_path = os.path.join(args.output_dir, f"{input_filename}_comparison_highres_epoch{epoch_number}.png")
    cv2.imwrite(comparison_high_res_save_path, comparison_high_res)
    print(f"💾 Comparação em alta resolução salva: {comparison_high_res_save_path}")
    
    # Criar comparação lado a lado (baixa resolução)
    comparison_low_res = np.hstack([input_resized, output_low_res])
    comparison_low_res_save_path = os.path.join(args.output_dir, f"{input_filename}_comparison_lowres_epoch{epoch_number}.png")
    cv2.imwrite(comparison_low_res_save_path, comparison_low_res)
    print(f"💾 Comparação em baixa resolução salva: {comparison_low_res_save_path}")
    
    # Processar mapa de atenção
    attention_map = attention.squeeze().cpu().numpy()
    attention_map = (attention_map * 255).astype(np.uint8)
    
    # Aplicar colormap ao mapa de atenção
    attention_colored = cv2.applyColorMap(attention_map, cv2.COLORMAP_JET)
    
    # Salvar mapa de atenção em baixa resolução
    attention_low_res_save_path = os.path.join(args.output_dir, f"{input_filename}_attention_lowres_epoch{epoch_number}.png")
    cv2.imwrite(attention_low_res_save_path, attention_colored)
    print(f"💾 Mapa de atenção em baixa resolução salvo: {attention_low_res_save_path}")
    
    # Upscalar mapa de atenção para alta resolução usando cubic convolution
    attention_high_res = cv2.resize(attention_colored, (original_width, original_height), interpolation=cv2.INTER_CUBIC)
    attention_high_res_save_path = os.path.join(args.output_dir, f"{input_filename}_attention_highres_epoch{epoch_number}.png")
    cv2.imwrite(attention_high_res_save_path, attention_high_res)
    print(f"💾 Mapa de atenção em alta resolução (cubic) salvo: {attention_high_res_save_path}")
    
    # Criar overlay do mapa de atenção na imagem original
    attention_overlay = cv2.addWeighted(input_img, 0.7, attention_high_res, 0.3, 0)
    attention_overlay_save_path = os.path.join(args.output_dir, f"{input_filename}_attention_overlay_epoch{epoch_number}.png")
    cv2.imwrite(attention_overlay_save_path, attention_overlay)
    print(f"💾 Overlay do mapa de atenção salvo: {attention_overlay_save_path}")
    
    print("✅ Teste concluído com sucesso!")
    print(f"📁 Resultados salvos em: {args.output_dir}")
    print(f"📊 Resolução original: {original_width}x{original_height}")
    print(f"📊 Resolução de processamento: 320x240")
    
    print("\n🎉 Teste executado com sucesso!")

if __name__ == '__main__':
    main() 