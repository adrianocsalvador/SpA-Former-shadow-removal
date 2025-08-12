#!/usr/bin/env python3

import sys
import os
# Adicionar o diretório raiz ao path
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

def show(img):
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def predict(args):

    gpu_manage(args)
    ### MODELS LOAD ###
    print('===> Loading models')

    gen = Generator(gpu_ids=args.gpu_ids)

    # Se um modelo pré-treinado foi fornecido, carregue-o
    if args.pretrained and args.pretrained != "none":
        try:
            param = torch.load(args.pretrained)
            gen.load_state_dict(param)
            print(f'✓ Modelo carregado: {args.pretrained}')
        except Exception as e:
            print(f'⚠️ Erro ao carregar modelo: {e}')
            print('🔄 Usando modelo inicializado aleatoriamente para teste...')
    else:
        print('🔄 Usando modelo inicializado aleatoriamente para teste...')

    if args.cuda:
        gen = gen.cuda(0)

    print ('<=== Model loaded')

    print('===> Loading test image')
    img = cv2.imread(args.test_filepath, 1).astype(np.float32)
    img = img / 255
    img = img.transpose(2, 0, 1)
    img = img[None]
    print ('<=== test image loaded')

    with torch.no_grad():
        x = torch.from_numpy(img)
        if args.cuda:
            x = x.cuda()
        
        print('===> Removing shadows...')
        start_time = time.time()
        att, out = gen(x)
        print('<=== finish! %.3fs cost.' % (time.time()-start_time))

        x_ = x.cpu().numpy()[0]
        x_rgb = x_ * 255
        x_rgb = x_rgb.transpose(1, 2, 0).astype('uint8')
        out_ = out.cpu().numpy()[0]
        out_rgb = np.clip(out_[:3], 0, 1) * 255
        out_rgb = out_rgb.transpose(1, 2, 0).astype('uint8')
        att_ = att.cpu().numpy()[0] * 255
        att_heatmap = heatmap(att_.astype('uint8'))[0]
        att_heatmap = att_heatmap.transpose(1, 2, 0)

        allim = np.hstack((x_rgb, out_rgb, att_heatmap))
        show(allim)

        # Salvar resultado
        output_path = args.test_filepath.replace('.jpeg', '_result.jpg').replace('.jpg', '_result.jpg').replace('.png', '_result.png')
        cv2.imwrite(output_path, allim)
        print(f'💾 Resultado salvo em: {output_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_filepath', type=str, required=True)
    parser.add_argument('--pretrained', type=str, default="none", help="Caminho para modelo pré-treinado (ou 'none' para modelo aleatório)")
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--gpu_ids', type=int, default=[0])
    parser.add_argument('--manualSeed', type=int, default=0)
    args = parser.parse_args()

    predict(args) 