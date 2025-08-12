#!/usr/bin/env python3

from PIL import Image
import argparse
import os

def resize_image(input_path, output_path, size=(640, 480)):
    img = Image.open(input_path)
    img_resized = img.resize(size, Image.LANCZOS)
    img_resized.save(output_path)
    print(f'Imagem redimensionada salva em: {output_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Redimensiona uma imagem para 640x480')
    parser.add_argument('--input', type=str, required=True, help='Caminho da imagem original')
    parser.add_argument('--output', type=str, required=True, help='Caminho para salvar a imagem redimensionada')
    args = parser.parse_args()

    resize_image(args.input, args.output) 