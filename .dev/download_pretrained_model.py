#!/usr/bin/env python3

import requests
import os
import sys

def download_file(url, filename):
    """Download a file from URL"""
    try:
        print(f"Baixando {filename}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192
        downloaded = 0
        
        with open(filename, 'wb') as f:
            for data in response.iter_content(block_size):
                f.write(data)
                downloaded += len(data)
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    print(f"\rProgresso: {percent:.1f}%", end='', flush=True)
        
        print(f"\n✅ {filename} baixado com sucesso!")
        return True
        
    except Exception as e:
        print(f"\n❌ Erro ao baixar {filename}: {e}")
        return False

def main():
    # Criar pasta se não existir
    os.makedirs("pretrained_models", exist_ok=True)
    
    # URLs de modelos pré-treinados (tentativas)
    model_urls = [
        # Tentativa 1: Hugging Face (se disponível)
        "https://huggingface.co/datasets/SpA-Former/models/resolve/main/gen_model_epoch_200.pth",
        
        # Tentativa 2: GitHub releases (se disponível)
        "https://github.com/zhangbaijin/SpA-Former-shadow-removal/releases/download/v1.0/gen_model_epoch_200.pth",
        
        # Tentativa 3: Outros repositórios
        "https://github.com/SpA-Former/models/raw/main/gen_model_epoch_200.pth"
    ]
    
    output_file = "pretrained_models/gen_model_epoch_200.pth"
    
    print("🔍 Tentando baixar modelo pré-treinado...")
    print("📋 Fontes disponíveis:")
    print("1. Google Drive: https://drive.google.com/drive/folders/1pxwwAfwnGKkLj-GAlkVCevbEQM4basgR?usp=sharing")
    print("2. Baidu Drive: https://pan.baidu.com/s/1slny1G_9WuxBcoyw5eKUVA (提取码：rpis)")
    print("")
    
    # Tentar baixar automaticamente
    success = False
    for i, url in enumerate(model_urls, 1):
        print(f"🔄 Tentativa {i}: {url}")
        if download_file(url, output_file):
            success = True
            break
        print("")
    
    if not success:
        print("❌ Não foi possível baixar automaticamente.")
        print("📥 Faça o download manual:")
        print("1. Acesse: https://drive.google.com/drive/folders/1pxwwAfwnGKkLj-GAlkVCevbEQM4basgR?usp=sharing")
        print("2. Baixe o arquivo 'gen_model_epoch_200.pth'")
        print("3. Coloque na pasta 'pretrained_models/'")
        print("")
        print("🎯 Comando para testar após baixar:")
        print(f"python .dev/test_demo.py --test_filepath \"/home/consultoria/Desktop/shadow_test1/20250715_01_00017_PAN.jpeg\" --pretrained {output_file} --cuda")
    else:
        print("🎯 Para testar o modelo:")
        print(f"python .dev/test_demo.py --test_filepath \"/home/consultoria/Desktop/shadow_test1/20250715_01_00017_PAN.jpeg\" --pretrained {output_file} --cuda")

if __name__ == "__main__":
    main() 