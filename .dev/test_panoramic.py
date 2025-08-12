#!/usr/bin/env python3

import sys
import os
import argparse
import cv2
import numpy as np
import time

# Adicionar o diretório raiz ao path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image
import torch
from torch.autograd import Variable

from utils import gpu_manage, heatmap
from SpA_Former import Generator

def resize_panoramic_for_test(input_path, target_size=(640, 480)):
    """Redimensiona imagem panorâmica para teste"""
    with Image.open(input_path) as img:
        # Calcular proporção para manter aspect ratio
        img_ratio = img.width / img.height
        target_ratio = target_size[0] / target_size[1]
        
        if img_ratio > target_ratio:
            # Imagem mais larga que o target - ajustar largura
            new_width = target_size[0]
            new_height = int(target_size[0] / img_ratio)
        else:
            # Imagem mais alta que o target - ajustar altura
            new_height = target_size[1]
            new_width = int(target_size[1] * img_ratio)
        
        # Redimensionar
        img_resized = img.resize((new_width, new_height), Image.LANCZOS)
        
        # Criar imagem final com padding se necessário
        final_img = Image.new('RGB', target_size, (0, 0, 0))
        
        # Centralizar a imagem redimensionada
        x_offset = (target_size[0] - new_width) // 2
        y_offset = (target_size[1] - new_height) // 2
        
        final_img.paste(img_resized, (x_offset, y_offset))
        return final_img

def test_panoramic_image(args):
    """Testa modelo com imagem panorâmica"""
    
    gpu_manage(args)
    
    print('===> Loading models')
    gen = Generator(gpu_ids=args.gpu_ids)

    # Carregar modelo pré-treinado se fornecido
    if args.pretrained and args.pretrained != "none":
        try:
            param = torch.load(args.pretrained)
            gen.load_state_dict(param)
            print(f'✓ Modelo carregado: {args.pretrained}')
        except Exception as e:
            print(f'⚠️ Erro ao carregar modelo: {e}')
            print('🔄 Usando modelo inicializado aleatoriamente...')
    else:
        print('🔄 Usando modelo inicializado aleatoriamente...')

    if args.cuda:
        gen = gen.cuda(0)

    print('<=== Model loaded')

    print('===> Loading panoramic image')
    
    # Redimensionar imagem panorâmica
    img_pil = resize_panoramic_for_test(args.test_filepath)
    img_array = np.array(img_pil).astype(np.float32)
    img = img_array / 255.0
    img = img.transpose(2, 0, 1)
    img = img[None]
    
    print(f'<=== Imagem carregada e redimensionada: {img_pil.size[0]}x{img_pil.size[1]}')

    with torch.no_grad():
        x = torch.from_numpy(img)
        if args.cuda:
            x = x.cuda()
        
        print('===> Removing shadows from panoramic image...')
        start_time = time.time()
        att, out = gen(x)
        print(f'<=== finish! {time.time()-start_time:.3f}s cost.')

        # Processar resultados
        x_ = x.cpu().numpy()[0]
        x_rgb = x_ * 255
        x_rgb = x_rgb.transpose(1, 2, 0).astype('uint8')
        
        out_ = out.cpu().numpy()[0]
        out_rgb = np.clip(out_[:3], 0, 1) * 255
        out_rgb = out_rgb.transpose(1, 2, 0).astype('uint8')
        
        att_ = att.cpu().numpy()[0] * 255
        att_heatmap = heatmap(att_.astype('uint8'))[0]
        att_heatmap = att_heatmap.transpose(1, 2, 0)

        # Combinar imagens
        allim = np.hstack((x_rgb, out_rgb, att_heatmap))
        
        # Salvar resultado
        output_path = args.test_filepath.replace('.jpeg', '_panoramic_result.jpg').replace('.jpg', '_panoramic_result.jpg').replace('.png', '_panoramic_result.png')
        cv2.imwrite(output_path, cv2.cvtColor(allim, cv2.COLOR_RGB2BGR))
        print(f'💾 Resultado salvo em: {output_path}')
        
        # Mostrar resultado se solicitado
        if args.show:
            cv2.imshow('Panoramic Shadow Removal - Original | Result | Attention', cv2.cvtColor(allim, cv2.COLOR_RGB2BGR))
            cv2.waitKey(0)
            cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Testar SpA-Former com imagem panorâmica')
    parser.add_argument('--test_filepath', type=str, required=True,
                       help='Caminho para imagem panorâmica de teste')
    parser.add_argument('--pretrained', type=str, default="none",
                       help='Caminho para modelo pré-treinado (ou "none" para modelo aleatório)')
    parser.add_argument('--cuda', action='store_true',
                       help='Usar GPU CUDA')
    parser.add_argument('--show', action='store_true',
                       help='Mostrar resultado na tela')
    parser.add_argument('--gpu_ids', type=int, default=[0])
    parser.add_argument('--manualSeed', type=int, default=0)
    
    args = parser.parse_args()

    test_panoramic_image(args) 