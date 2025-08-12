#!/usr/bin/env python3
"""
Script para iniciar treinamento com 1024 imagens usando modelo pré-treinado
"""

import os
import sys
import yaml
import argparse

def main():
    parser = argparse.ArgumentParser(description='Iniciar treinamento com 1024 imagens e modelo pré-treinado')
    parser.add_argument('--config', type=str, 
                       default='/mnt/48EC7EE9EC7ED0A4/Teste_Sombra/dataset_v2/config_panoramic.yml',
                       help='Caminho para o arquivo de configuração')
    parser.add_argument('--pretrained_model', type=str,
                       default='results_panoramic/models/gen_model_epoch_34.pth',
                       help='Caminho para o modelo pré-treinado')
    parser.add_argument('--max_images', type=int, default=1024,
                       help='Número máximo de imagens para treinar')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Número de épocas para treinar')
    args = parser.parse_args()
    
    print("🚀 Iniciando treinamento com 1024 imagens e modelo pré-treinado...")
    print(f"📂 Configuração: {args.config}")
    print(f"🤖 Modelo pré-treinado: {args.pretrained_model}")
    print(f"📊 Imagens: {args.max_images}")
    print(f"📊 Épocas: {args.epochs}")
    
    # Verificar se o arquivo de configuração existe
    if not os.path.exists(args.config):
        print(f"❌ Arquivo de configuração não encontrado: {args.config}")
        return False
    
    # Verificar se o modelo pré-treinado existe
    if not os.path.exists(args.pretrained_model):
        print(f"❌ Modelo pré-treinado não encontrado: {args.pretrained_model}")
        return False
    
    # Carregar configuração
    with open(args.config, 'r', encoding='UTF-8') as f:
        config_dict = yaml.safe_load(f)
    
    # Atualizar configuração para o novo treinamento
    config_dict['epoch'] = args.epochs
    config_dict['n_data'] = args.max_images
    
    # Salvar configuração atualizada
    updated_config_path = args.config.replace('.yml', '_1024.yml')
    with open(updated_config_path, 'w', encoding='UTF-8') as f:
        yaml.dump(config_dict, f, default_flow_style=False)
    
    print(f"📄 Configuração atualizada salva: {updated_config_path}")
    
    # Comando para executar o treinamento
    cmd = f"python3 .dev/real_train.py --config {updated_config_path} --max_images {args.max_images} --pretrained_model {args.pretrained_model}"
    
    print("\n🎯 Comando para executar:")
    print(f"   {cmd}")
    
    # Perguntar se deve executar
    response = input("\n❓ Deseja executar o treinamento agora? (s/n): ").lower().strip()
    
    if response in ['s', 'sim', 'y', 'yes']:
        print("\n🚀 Executando treinamento...")
        os.system(cmd)
    else:
        print("\n✅ Comando preparado. Execute manualmente quando estiver pronto.")
        print(f"   {cmd}")
    
    return True

if __name__ == '__main__':
    main() 