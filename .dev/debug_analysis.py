#!/usr/bin/env python3

import re

def debug_parse():
    with open('./results_panoramic/real_training_log.txt', 'r') as f:
        content = f.read()
    
    # Padrão para encontrar as métricas
    pattern = r'Iteração (\d+): Loss_D=([\d.]+), Loss_G=([\d.]+)'
    matches = re.findall(pattern, content)
    
    print(f"Total de matches encontrados: {len(matches)}")
    
    # Verificar as primeiras e últimas iterações
    print("\nPrimeiras 5 iterações:")
    for i in range(min(5, len(matches))):
        print(f"  {matches[i]}")
    
    print("\nÚltimas 5 iterações:")
    for i in range(max(0, len(matches)-5), len(matches)):
        print(f"  {matches[i]}")
    
    # Verificar se há padrão nas iterações
    iterations = [int(match[0]) for match in matches]
    print(f"\nNúmero da primeira iteração: {iterations[0]}")
    print(f"Número da última iteração: {iterations[-1]}")
    
    # Verificar se há 10 iterações por época
    print(f"\nVerificando se há 10 iterações por época...")
    for i in range(0, len(matches), 10):
        epoch_iterations = iterations[i:i+10]
        print(f"Época {i//10 + 1}: {epoch_iterations}")

if __name__ == '__main__':
    debug_parse() 