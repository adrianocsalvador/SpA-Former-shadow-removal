#!/usr/bin/env python3

import os
import shutil
import argparse
from pathlib import Path

def setup_dataset_structure(source_dir, target_dir, output_dir):
    """
    Configura a estrutura de dataset para o SpA-Former
    
    Args:
        source_dir: Pasta com imagens COM sombra
        target_dir: Pasta com imagens SEM sombra (ground truth)
        output_dir: Pasta de saída para organizar o dataset
    """
    
    # Criar estrutura de pastas
    train_input_dir = os.path.join(output_dir, "ISTD", "train", "input")
    train_target_dir = os.path.join(output_dir, "ISTD", "train", "target")
    test_input_dir = os.path.join(output_dir, "ISTD", "test", "input")
    test_target_dir = os.path.join(output_dir, "ISTD", "test", "target")
    
    os.makedirs(train_input_dir, exist_ok=True)
    os.makedirs(train_target_dir, exist_ok=True)
    os.makedirs(test_input_dir, exist_ok=True)
    os.makedirs(test_target_dir, exist_ok=True)
    
    print(f"📁 Criando estrutura de pastas em: {output_dir}")
    
    # Listar arquivos
    source_files = [f for f in os.listdir(source_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    target_files = [f for f in os.listdir(target_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"📊 Encontrados {len(source_files)} arquivos com sombra")
    print(f"📊 Encontrados {len(target_files)} arquivos sem sombra")
    
    # Verificar correspondência
    source_names = {os.path.splitext(f)[0] for f in source_files}
    target_names = {os.path.splitext(f)[0] for f in target_files}
    matching_names = source_names.intersection(target_names)
    
    print(f"✅ {len(matching_names)} pares de imagens correspondentes encontrados")
    
    if len(matching_names) == 0:
        print("❌ Nenhum par correspondente encontrado!")
        print("💡 Certifique-se de que as imagens tenham nomes correspondentes")
        return False
    
    # Dividir em treino e teste (80% treino, 20% teste)
    matching_list = list(matching_names)
    split_idx = int(len(matching_list) * 0.8)
    train_names = matching_list[:split_idx]
    test_names = matching_list[split_idx:]
    
    print(f"🎓 {len(train_names)} imagens para treino")
    print(f"🧪 {len(test_names)} imagens para teste")
    
    # Copiar arquivos para treino
    for name in train_names:
        # Encontrar extensões
        source_ext = next(ext for f in source_files if os.path.splitext(f)[0] == name)
        target_ext = next(ext for f in target_files if os.path.splitext(f)[0] == name)
        
        # Copiar arquivos
        shutil.copy2(
            os.path.join(source_dir, name + os.path.splitext(source_ext)[1]),
            os.path.join(train_input_dir, f"{name}.png")
        )
        shutil.copy2(
            os.path.join(target_dir, name + os.path.splitext(target_ext)[1]),
            os.path.join(train_target_dir, f"{name}.png")
        )
    
    # Copiar arquivos para teste
    for name in test_names:
        # Encontrar extensões
        source_ext = next(ext for f in source_files if os.path.splitext(f)[0] == name)
        target_ext = next(ext for f in target_files if os.path.splitext(f)[0] == name)
        
        # Copiar arquivos
        shutil.copy2(
            os.path.join(source_dir, name + os.path.splitext(source_ext)[1]),
            os.path.join(test_input_dir, f"{name}.png")
        )
        shutil.copy2(
            os.path.join(target_dir, name + os.path.splitext(target_ext)[1]),
            os.path.join(test_target_dir, f"{name}.png")
        )
    
    print("✅ Dataset configurado com sucesso!")
    print(f"📁 Estrutura criada em: {output_dir}")
    
    return True

def create_config_file(output_dir, num_epochs=100, batch_size=4):
    """Cria arquivo de configuração personalizado"""
    
    config_content = f"""# Configuração para treinamento personalizado
datasets_dir: {output_dir}/ISTD/train
valset_dir: {output_dir}/ISTD/test
train_list: train_list.txt
test_list: 
validation_list: val_list.txt
out_dir: ./results_custom

cuda: True
gpu_ids: [0]

train_size: 2
val_size: 0
batchsize: {batch_size}
validation_batchsize: {batch_size}
epoch: {num_epochs}
n_data: 433  # Será atualizado automaticamente
width: 480
height: 640
threads: 4

lr: 0.0004
beta1: 0.5
lamb: 100
minimax: 1

gen_init: 
dis_init: 
in_ch: 3
out_ch: 3

manualSeed: 0
snapshot_interval: 1
"""
    
    config_path = os.path.join(output_dir, "config_custom.yml")
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    print(f"⚙️ Arquivo de configuração criado: {config_path}")
    return config_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Configurar dataset para treinamento SpA-Former')
    parser.add_argument('--source_dir', type=str, required=True, 
                       help='Pasta com imagens COM sombra')
    parser.add_argument('--target_dir', type=str, required=True, 
                       help='Pasta com imagens SEM sombra (ground truth)')
    parser.add_argument('--output_dir', type=str, default='./data_custom',
                       help='Pasta de saída para o dataset organizado')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Número de épocas para treinamento')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Tamanho do batch')
    
    args = parser.parse_args()
    
    print("🚀 Configurando dataset para treinamento SpA-Former")
    print(f"📂 Imagens com sombra: {args.source_dir}")
    print(f"📂 Imagens sem sombra: {args.target_dir}")
    print(f"📂 Saída: {args.output_dir}")
    
    if setup_dataset_structure(args.source_dir, args.target_dir, args.output_dir):
        config_path = create_config_file(args.output_dir, args.epochs, args.batch_size)
        
        print("\n🎯 Próximos passos:")
        print(f"1. Verifique a estrutura em: {args.output_dir}")
        print(f"2. Ajuste a configuração em: {config_path}")
        print(f"3. Execute o treinamento:")
        print(f"   python train.py --config {config_path}")
        print("\n💡 Dicas:")
        print("- Use imagens de tamanho similar (640x480 recomendado)")
        print("- Quanto mais dados, melhor o resultado")
        print("- Treinamento pode levar várias horas dependendo do hardware")
    else:
        print("❌ Falha na configuração do dataset") 