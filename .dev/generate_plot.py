#!/usr/bin/env python3
"""
Script para gerar gráfico de evolução do treinamento
"""

import os
import re
import matplotlib
matplotlib.use('Agg')  # Para funcionar sem display
import matplotlib.pyplot as plt
import numpy as np

def parse_training_logs(log_file_path):
    """Extrai dados do arquivo de log de treinamento"""
    epochs = []
    loss_d_values = []
    loss_g_values = []
    
    if not os.path.exists(log_file_path):
        print(f"❌ Arquivo de log não encontrado: {log_file_path}")
        return None
    
    with open(log_file_path, 'r') as f:
        for line in f:
            # Padrão: Época X - Loss_D: Y, Loss_G: Z (Tempo: Ws)
            match = re.search(r'Época (\d+) - Loss_D: ([\d.]+), Loss_G: ([\d.]+)', line)
            if match:
                epoch = int(match.group(1))
                loss_d = float(match.group(2))
                loss_g = float(match.group(3))
                
                epochs.append(epoch)
                loss_d_values.append(loss_d)
                loss_g_values.append(loss_g)
    
    if not epochs:
        print("❌ Nenhum dado encontrado no arquivo de log")
        return None
    
    return {
        'epochs': epochs,
        'loss_d': loss_d_values,
        'loss_g': loss_g_values
    }

def create_training_plots(data, output_dir):
    """Cria gráfico de evolução do treinamento"""
    if data is None:
        print("❌ Dados inválidos para gerar gráfico")
        return False
    
    epochs = data['epochs']
    loss_d = data['loss_d']
    loss_g = data['loss_g']
    
    plt.figure(figsize=(12, 8))
    
    # Gráfico principal
    plt.subplot(2, 1, 1)
    plt.plot(epochs, loss_d, 'b-', label='Loss_D (Discriminador)', linewidth=2)
    plt.plot(epochs, loss_g, 'r-', label='Loss_G (Gerador)', linewidth=2)
    plt.xlabel('Época')
    plt.ylabel('Loss')
    plt.title('Evolução do Treinamento - SpA-Former')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Gráfico de zoom (últimas 20 épocas)
    plt.subplot(2, 1, 2)
    if len(epochs) > 20:
        start_idx = len(epochs) - 20
        plt.plot(epochs[start_idx:], loss_d[start_idx:], 'b-', label='Loss_D (Discriminador)', linewidth=2)
        plt.plot(epochs[start_idx:], loss_g[start_idx:], 'r-', label='Loss_G (Gerador)', linewidth=2)
        plt.xlabel('Época')
        plt.ylabel('Loss')
        plt.title('Zoom - Últimas 20 Épocas')
        plt.legend()
        plt.grid(True, alpha=0.3)
    else:
        plt.plot(epochs, loss_d, 'b-', label='Loss_D (Discriminador)', linewidth=2)
        plt.plot(epochs, loss_g, 'r-', label='Loss_G (Gerador)', linewidth=2)
        plt.xlabel('Época')
        plt.ylabel('Loss')
        plt.title('Evolução Completa')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Salvar gráfico
    plot_path = os.path.join(output_dir, 'training_evolution.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Gráfico salvo: {plot_path}")
    return True

def main():
    """Função principal"""
    log_file = "results_panoramic/training_log.txt"
    output_dir = "results_panoramic"
    
    print("📊 Gerando gráfico de evolução do treinamento...")
    
    # Extrair dados do log
    data = parse_training_logs(log_file)
    if data is None:
        return False
    
    print(f"📈 Dados extraídos: {len(data['epochs'])} épocas")
    print(f"   Loss_D: {data['loss_d'][0]:.4f} → {data['loss_d'][-1]:.4f}")
    print(f"   Loss_G: {data['loss_g'][0]:.4f} → {data['loss_g'][-1]:.4f}")
    
    # Gerar gráfico
    success = create_training_plots(data, output_dir)
    
    if success:
        print("✅ Gráfico gerado com sucesso!")
    else:
        print("❌ Falha ao gerar gráfico")
    
    return success

if __name__ == "__main__":
    main() 