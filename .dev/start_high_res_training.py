#!/usr/bin/env python3

import os
import sys
import subprocess
import argparse
import yaml
from pathlib import Path

def update_config_for_training(config_path, max_images=None, pretrained_model=None):
    """Atualiza configuração para treinamento"""
    
    # Ler configuração atual
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Atualizar parâmetros
    if max_images:
        config['n_data'] = max_images
        print(f"📊 Limitando dataset para {max_images} imagens")
    
    # Ajustar batch size para alta resolução
    config['batchsize'] = 1  # Reduzido para evitar OOM
    config['validation_batchsize'] = 1
    
    # Learning rate ajustado para alta resolução
    config['lr'] = 0.0001  # Reduzido para estabilidade
    
    # Salvar configuração atualizada
    temp_config_path = config_path.replace('.yml', '_temp.yml')
    with open(temp_config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"⚙️ Configuração atualizada salva: {temp_config_path}")
    return temp_config_path

def start_training(config_path, max_images=None, pretrained_model=None):
    """Inicia o treinamento"""
    
    print("🚀 Iniciando Treinamento com Alta Resolução")
    print("=" * 50)
    
    # Atualizar configuração
    temp_config = update_config_for_training(config_path, max_images, pretrained_model)
    
    # Construir comando
    cmd = [
        'python3', '.dev/real_train.py',
        '--config', temp_config
    ]
    
    if max_images:
        cmd.extend(['--max_images', str(max_images)])
    
    if pretrained_model:
        cmd.extend(['--pretrained_model', pretrained_model])
    
    print(f"🔧 Comando: {' '.join(cmd)}")
    print(f"📁 Config: {temp_config}")
    print(f"📊 Máximo de imagens: {max_images if max_images else 'Todas'}")
    print(f"🎯 Modelo pré-treinado: {pretrained_model if pretrained_model else 'Nenhum'}")
    
    # Executar treinamento
    try:
        print("\n🚀 Iniciando treinamento...")
        result = subprocess.run(cmd, check=True)
        print("✅ Treinamento concluído com sucesso!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Erro durante treinamento: {e}")
        return False
    except KeyboardInterrupt:
        print("\n⏹️ Treinamento interrompido pelo usuário")
        return False
    
    return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Iniciar treinamento com alta resolução')
    parser.add_argument('--config', type=str, required=True,
                       help='Caminho para arquivo de configuração')
    parser.add_argument('--max_images', type=int, default=None,
                       help='Número máximo de imagens para treinar')
    parser.add_argument('--pretrained_model', type=str, default=None,
                       help='Caminho para modelo pré-treinado')
    
    args = parser.parse_args()
    
    # Verificar se arquivo de configuração existe
    if not os.path.exists(args.config):
        print(f"❌ Arquivo de configuração não encontrado: {args.config}")
        sys.exit(1)
    
    # Iniciar treinamento
    success = start_training(args.config, args.max_images, args.pretrained_model)
    
    if success:
        print("\n🎉 Treinamento concluído!")
        print("📁 Verifique os resultados em: ./results_high_res")
    else:
        print("\n❌ Treinamento falhou!")
        sys.exit(1) 