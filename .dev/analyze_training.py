#!/usr/bin/env python3

import os
import sys
import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def parse_training_logs(log_file_path):
    """
    Analisa os logs de treinamento e extrai as métricas
    """
    epochs = []
    loss_d_values = []
    loss_g_values = []
    loss_g_l1_values = []
    
    print(f"📊 Analisando logs: {log_file_path}")
    
    if not os.path.exists(log_file_path):
        print(f"❌ Arquivo de log não encontrado: {log_file_path}")
        return None
    
    with open(log_file_path, 'r') as f:
        content = f.read()
    
    # Padrão para encontrar as métricas
    pattern = r'Iteração (\d+): Loss_D=([\d.]+), Loss_G=([\d.]+)'
    matches = re.findall(pattern, content)
    
    if not matches:
        print("❌ Nenhuma métrica encontrada nos logs")
        return None
    
    print(f"✅ Encontradas {len(matches)} iterações com métricas")
    
    # Agrupar por época (cada época tem 2 iterações: 5 e 10)
    current_epoch = 1
    epoch_loss_d = []
    epoch_loss_g = []
    
    for i, (iteration, loss_d, loss_g) in enumerate(matches):
        epoch_loss_d.append(float(loss_d))
        epoch_loss_g.append(float(loss_g))
        
        # A cada 2 iterações, finaliza uma época
        if (i + 1) % 2 == 0:
            epochs.append(current_epoch)
            loss_d_values.append(np.mean(epoch_loss_d))
            loss_g_values.append(np.mean(epoch_loss_g))
            
            # Extrair Loss_G_L1 se disponível
            loss_g_l1 = loss_g  # Por enquanto, usamos o Loss_G total
            loss_g_l1_values.append(loss_g_l1)
            
            current_epoch += 1
            epoch_loss_d = []
            epoch_loss_g = []
    
    print(f"📊 Processadas {len(epochs)} épocas completas")
    
    return {
        'epochs': epochs,
        'loss_d': loss_d_values,
        'loss_g': loss_g_values,
        'loss_g_l1': loss_g_l1_values
    }

def create_training_plots(data, output_dir):
    """
    Cria gráficos de evolução do treinamento
    """
    if not data:
        print("❌ Nenhum dado para plotar")
        return
    
    # Configurar estilo dos gráficos
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Evolução do Treinamento SpA-Former', fontsize=16, fontweight='bold')
    
    epochs = data['epochs']
    
    # Gráfico 1: Loss_D vs Época
    axes[0, 0].plot(epochs, data['loss_d'], 'b-', linewidth=2, label='Loss_D')
    axes[0, 0].set_title('Evolução da Loss do Discriminador (Loss_D)', fontweight='bold')
    axes[0, 0].set_xlabel('Época')
    axes[0, 0].set_ylabel('Loss_D')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # Gráfico 2: Loss_G vs Época
    axes[0, 1].plot(epochs, data['loss_g'], 'r-', linewidth=2, label='Loss_G')
    axes[0, 1].set_title('Evolução da Loss do Gerador (Loss_G)', fontweight='bold')
    axes[0, 1].set_xlabel('Época')
    axes[0, 1].set_ylabel('Loss_G')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # Gráfico 3: Loss_D e Loss_G juntas
    axes[1, 0].plot(epochs, data['loss_d'], 'b-', linewidth=2, label='Loss_D')
    axes[1, 0].plot(epochs, data['loss_g'], 'r-', linewidth=2, label='Loss_G')
    axes[1, 0].set_title('Comparação Loss_D vs Loss_G', fontweight='bold')
    axes[1, 0].set_xlabel('Época')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    # Gráfico 4: Análise de convergência
    # Calcular médias móveis para suavizar
    window = 5
    if len(data['loss_d']) > window:
        loss_d_smooth = np.convolve(data['loss_d'], np.ones(window)/window, mode='valid')
        loss_g_smooth = np.convolve(data['loss_g'], np.ones(window)/window, mode='valid')
        epochs_smooth = epochs[window-1:]
        
        axes[1, 1].plot(epochs_smooth, loss_d_smooth, 'b-', linewidth=2, label='Loss_D (suavizada)')
        axes[1, 1].plot(epochs_smooth, loss_g_smooth, 'r-', linewidth=2, label='Loss_G (suavizada)')
        axes[1, 1].set_title('Tendência de Convergência (Média Móvel)', fontweight='bold')
        axes[1, 1].set_xlabel('Época')
        axes[1, 1].set_ylabel('Loss (Suavizada)')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()
    else:
        axes[1, 1].text(0.5, 0.5, 'Dados insuficientes\npara análise de convergência', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Análise de Convergência', fontweight='bold')
    
    plt.tight_layout()
    
    # Salvar gráfico
    output_path = os.path.join(output_dir, 'training_evolution.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📊 Gráfico salvo: {output_path}")
    
    # Mostrar estatísticas
    print("\n📈 Estatísticas do Treinamento:")
    print(f"   Total de épocas: {len(epochs)}")
    print(f"   Loss_D inicial: {data['loss_d'][0]:.4f}")
    print(f"   Loss_D final: {data['loss_d'][-1]:.4f}")
    print(f"   Loss_G inicial: {data['loss_g'][0]:.4f}")
    print(f"   Loss_G final: {data['loss_g'][-1]:.4f}")
    
    # Análise de convergência
    if len(data['loss_d']) > 10:
        recent_loss_d = data['loss_d'][-10:]
        recent_loss_g = data['loss_g'][-10:]
        
        print(f"\n🎯 Análise das últimas 10 épocas:")
        print(f"   Loss_D média: {np.mean(recent_loss_d):.4f} ± {np.std(recent_loss_d):.4f}")
        print(f"   Loss_G média: {np.mean(recent_loss_g):.4f} ± {np.std(recent_loss_g):.4f}")
        
        # Verificar se está convergindo
        if np.std(recent_loss_d) < 0.01 and np.std(recent_loss_g) < 1.0:
            print("   ✅ Treinamento parece estar convergindo!")
        else:
            print("   ⚠️ Treinamento pode precisar de mais épocas")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Analisar logs de treinamento do SpA-Former')
    parser.add_argument('--log_file', type=str, 
                       help='Caminho para o arquivo de log (opcional)')
    parser.add_argument('--output_dir', type=str, default='./results_panoramic',
                       help='Diretório de saída para os gráficos')
    
    args = parser.parse_args()
    
    # Se não especificado, procurar por logs recentes
    if not args.log_file:
        # Procurar por logs no diretório de resultados
        results_dir = args.output_dir
        if os.path.exists(results_dir):
            log_files = list(Path(results_dir).glob('*.log'))
            if log_files:
                args.log_file = str(log_files[-1])  # Usar o mais recente
                print(f"📄 Usando log mais recente: {args.log_file}")
            else:
                print("❌ Nenhum arquivo de log encontrado")
                print("💡 Execute o treinamento primeiro ou especifique --log_file")
                return
        else:
            print("❌ Diretório de resultados não encontrado")
            return
    
    # Analisar logs
    data = parse_training_logs(args.log_file)
    
    if data:
        # Criar gráficos
        os.makedirs(args.output_dir, exist_ok=True)
        create_training_plots(data, args.output_dir)
    else:
        print("❌ Falha na análise dos logs")

if __name__ == '__main__':
    import argparse
    main() 