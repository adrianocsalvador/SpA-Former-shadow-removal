#!/usr/bin/env python3

import os
import sys
import yaml
import argparse
import subprocess
from pathlib import Path

# Adicionar o diretório raiz ao path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def update_config_with_image_count(config_path):
    """
    Atualiza a configuração com o número correto de imagens
    """
    print(f"📊 Atualizando configuração: {config_path}")
    
    # Ler configuração atual
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Contar imagens de treino (usando train_A que é o padrão do data_manager.py)
    train_A_dir = os.path.join(config['datasets_dir'], 'train_A')
    if os.path.exists(train_A_dir):
        train_images = len([f for f in os.listdir(train_A_dir) if f.endswith('.png')])
        config['n_data'] = train_images
        print(f"✅ {train_images} imagens de treino encontradas")
    else:
        print(f"❌ Diretório de treino não encontrado: {train_A_dir}")
        return False
    
    # Verificar imagens de teste
    test_A_dir = os.path.join(config['valset_dir'], 'test_A')
    if os.path.exists(test_A_dir):
        test_images = len([f for f in os.listdir(test_A_dir) if f.endswith('.png')])
        print(f"✅ {test_images} imagens de teste encontradas")
    else:
        print(f"⚠️ Diretório de teste não encontrado: {test_A_dir}")
    
    # Ajustar batch size se necessário
    if train_images < config['batchsize']:
        new_batch_size = max(1, train_images // 10)
        print(f"⚠️ Batch size muito grande para {train_images} imagens")
        print(f"🔄 Reduzindo batch size de {config['batchsize']} para {new_batch_size}")
        config['batchsize'] = new_batch_size
        config['validation_batchsize'] = new_batch_size
    
    # Salvar configuração atualizada
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"✅ Configuração atualizada e salva")
    return True

def create_train_list(config_path):
    """
    Cria o arquivo train_list.txt necessário para o treinamento
    """
    print("📝 Criando train_list.txt...")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    train_A_dir = os.path.join(config['datasets_dir'], 'train_A')
    train_C_dir = os.path.join(config['datasets_dir'], 'train_C')
    
    train_list_path = os.path.join(os.path.dirname(config['datasets_dir']), 'train_list.txt')
    
    with open(train_list_path, 'w') as f:
        for filename in sorted(os.listdir(train_A_dir)):
            if filename.endswith('.png'):
                f.write(f"{filename}\n")
    
    print(f"✅ train_list.txt criado: {train_list_path}")
    return train_list_path

def start_training(config_path):
    """
    Inicia o treinamento do modelo
    """
    print("🚀 Iniciando treinamento do SpA-Former...")
    print(f"📂 Configuração: {config_path}")
    
    # Verificar se estamos no diretório correto
    if not os.path.exists('train.py'):
        print("❌ Erro: train.py não encontrado!")
        print("💡 Execute este script do diretório raiz do projeto SpA-Former")
        return False
    
    # Verificar se o modelo pré-treinado existe (opcional)
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if config.get('gen_init') and os.path.exists(config['gen_init']):
        print(f"✅ Modelo pré-treinado encontrado: {config['gen_init']}")
    else:
        print("⚠️ Nenhum modelo pré-treinado especificado - treinamento do zero")
    
    # Comando de treinamento usando nossa versão corrigida
    cmd = [
        sys.executable, '.dev/train_fixed.py',
        '--config', config_path
    ]
    
    print(f"🎯 Comando: {' '.join(cmd)}")
    print("\n" + "="*60)
    print("🚀 INICIANDO TREINAMENTO...")
    print("="*60)
    
    try:
        # Executar treinamento
        result = subprocess.run(cmd, check=True)
        print("\n✅ Treinamento concluído com sucesso!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Erro durante o treinamento: {e}")
        return False
    except KeyboardInterrupt:
        print("\n⚠️ Treinamento interrompido pelo usuário")
        return False

def main():
    parser = argparse.ArgumentParser(description='Iniciar treinamento do SpA-Former')
    parser.add_argument('--config', type=str, required=True,
                       help='Caminho para o arquivo de configuração')
    parser.add_argument('--skip_setup', action='store_true',
                       help='Pular configuração inicial (se já foi feita)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        print(f"❌ Arquivo de configuração não encontrado: {args.config}")
        return 1
    
    print("🎯 Preparando treinamento do SpA-Former")
    print(f"📂 Configuração: {args.config}")
    
    if not args.skip_setup:
        # Atualizar configuração
        if not update_config_with_image_count(args.config):
            return 1
        
        # Criar train_list.txt
        create_train_list(args.config)
    
    # Iniciar treinamento
    if start_training(args.config):
        print("\n🎉 Treinamento finalizado!")
        print("\n📚 Próximos passos:")
        print("1. Verifique os resultados em ./results_panoramic/")
        print("2. Teste o modelo treinado:")
        print("   python .dev/test_panoramic.py --test_filepath <imagem> --pretrained ./results_panoramic/gen_model_epoch_X.pth")
        return 0
    else:
        print("\n❌ Falha no treinamento")
        return 1

if __name__ == '__main__':
    sys.exit(main()) 