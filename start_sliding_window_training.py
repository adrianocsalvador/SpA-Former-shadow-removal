#!/usr/bin/env python3
"""
Script de inicialização para treinamento com janela deslizante
SpA-Former Shadow Removal com janela deslizante em tempo real
"""

import os
import sys
import argparse
import yaml
from attrdict import AttrMap

def main():
    parser = argparse.ArgumentParser(description='SpA-Former Sliding Window Training')
    parser.add_argument('--config', type=str, default='config_sliding_window.yml',
                       help='Caminho para arquivo de configuração')
    parser.add_argument('--window_size', type=int, nargs=2, default=[640, 480],
                       help='Tamanho da janela (width height)')
    parser.add_argument('--stride', type=int, nargs=2, default=[320, 240],
                       help='Stride da janela (width height)')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Tamanho do batch')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Número de épocas')
    parser.add_argument('--lr', type=float, default=0.0004,
                       help='Learning rate')
    parser.add_argument('--pretrained_gen', type=str, default='',
                       help='Caminho para modelo gerador pré-treinado')
    parser.add_argument('--pretrained_dis', type=str, default='',
                       help='Caminho para modelo discriminador pré-treinado')
    
    args = parser.parse_args()
    
    # Carregar configuração
    if not os.path.exists(args.config):
        print(f"Erro: Arquivo de configuração {args.config} não encontrado!")
        print("Criando arquivo de configuração padrão...")
        create_default_config(args.config)
    
    with open(args.config, 'r', encoding='UTF-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    # Atualizar configuração com argumentos da linha de comando
    config['window_width'] = args.window_size[0]
    config['window_height'] = args.window_size[1]
    config['stride_width'] = args.stride[0]
    config['stride_height'] = args.stride[1]
    config['batchsize'] = args.batch_size
    config['epoch'] = args.epochs
    config['lr'] = args.lr
    
    if args.pretrained_gen:
        config['gen_init'] = args.pretrained_gen
    if args.pretrained_dis:
        config['dis_init'] = args.pretrained_dis
    
    # Verificar se o dataset existe
    if not os.path.exists(config['datasets_dir']):
        print(f"Erro: Dataset não encontrado em {config['datasets_dir']}")
        print("Certifique-se de que o dataset ISTD está configurado corretamente.")
        return
    
    # Verificar se as pastas train_A e train_C existem
    train_a_dir = os.path.join(config['datasets_dir'], 'train_A')
    train_c_dir = os.path.join(config['datasets_dir'], 'train_C')
    
    if not os.path.exists(train_a_dir):
        print(f"Erro: Pasta train_A não encontrada em {train_a_dir}")
        return
    
    if not os.path.exists(train_c_dir):
        print(f"Erro: Pasta train_C não encontrada em {train_c_dir}")
        return
    
    print("=== Configuração do Treinamento com Janela Deslizante ===")
    print(f"Dataset: {config['datasets_dir']}")
    print(f"Tamanho da janela: {config['window_width']}x{config['window_height']}")
    print(f"Stride: {config['stride_width']}x{config['stride_height']}")
    print(f"Batch size: {config['batchsize']}")
    print(f"Épocas: {config['epoch']}")
    print(f"Learning rate: {config['lr']}")
    print(f"Output directory: {config['out_dir']}")
    
    if config['gen_init']:
        print(f"Gerador pré-treinado: {config['gen_init']}")
    if config['dis_init']:
        print(f"Discriminador pré-treinado: {config['dis_init']}")
    
    print("\nIniciando treinamento...")
    
    # Converter para AttrMap
    config = AttrMap(config)
    
    # Importar e executar treinamento
    try:
        from sliding_window_train import train_with_sliding_window
        import utils
        
        # Criar diretório de saída
        utils.make_manager()
        n_job = utils.job_increment()
        config.out_dir = os.path.join(config.out_dir, '{:06}'.format(n_job))
        os.makedirs(config.out_dir)
        print(f'Job number: {n_job:04d}')
        
        # Salvar configuração
        import shutil
        shutil.copyfile(args.config, os.path.join(config.out_dir, 'config.yml'))
        
        # Executar treinamento
        train_with_sliding_window(config)
        
    except ImportError as e:
        print(f"Erro ao importar módulos: {e}")
        print("Certifique-se de que todos os arquivos necessários estão presentes.")
    except Exception as e:
        print(f"Erro durante o treinamento: {e}")
        import traceback
        traceback.print_exc()

def create_default_config(config_path):
    """Cria arquivo de configuração padrão"""
    default_config = {
        'datasets_dir': './data/ISTD/train',
        'valset_dir': './data/ISTD/test',
        'train_list': 'train_list.txt',
        'test_list': '',
        'validation_list': 'val_list.txt',
        'out_dir': './results_sliding_window',
        'cuda': True,
        'gpu_ids': [0],
        'window_width': 640,
        'window_height': 480,
        'stride_width': 320,
        'stride_height': 240,
        'batchsize': 4,
        'validation_batchsize': 4,
        'epoch': 100,
        'threads': 4,
        'lr': 0.0004,
        'beta1': 0.5,
        'lamb': 100,
        'minimax': 1,
        'gen_init': '',
        'dis_init': '',
        'in_ch': 3,
        'out_ch': 3,
        'manualSeed': 42,
        'snapshot_interval': 5,
        'enable_data_augmentation': True,
        'random_crop': True,
        'horizontal_flip': True,
        'vertical_flip': False,
        'rotation_range': 10
    }
    
    with open(config_path, 'w') as f:
        yaml.dump(default_config, f, default_flow_style=False)
    
    print(f"Arquivo de configuração criado: {config_path}")

if __name__ == '__main__':
    main()
