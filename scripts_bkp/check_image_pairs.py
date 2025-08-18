#!/usr/bin/env python3
"""
Verificar pares correspondentes de imagens com e sem sombra
"""

import os
from pathlib import Path


def check_image_pairs():
    """Verifica se existem pares correspondentes de imagens"""
    
    base_dir = "/mnt/48EC7EE9EC7ED0A4/Teste_Sombra"
    shadow_dir = os.path.join(base_dir, "20250717_01/PAN")  # Imagens com sombra (.jpeg)
    no_shadow_dir = os.path.join(base_dir, "20250717_01/PAN_SOMBRA")  # Imagens sem sombra (.jpg)
    
    # Listar arquivos
    shadow_files = [f for f in os.listdir(shadow_dir) if f.endswith('.jpeg')]
    no_shadow_files = [f for f in os.listdir(no_shadow_dir) if f.endswith('.jpg')]
    
    print(f"📊 Total de arquivos:")
    print(f"   Com sombra (.jpeg): {len(shadow_files)}")
    print(f"   Sem sombra (.jpg): {len(no_shadow_files)}")
    
    # Extrair números dos nomes
    shadow_numbers = set()
    no_shadow_numbers = set()
    
    for filename in shadow_files:
        number = Path(filename).stem  # Remove extensão
        shadow_numbers.add(number)
    
    for filename in no_shadow_files:
        number = Path(filename).stem  # Remove extensão
        no_shadow_numbers.add(number)
    
    print(f"\n📋 Números únicos:")
    print(f"   Com sombra: {len(shadow_numbers)}")
    print(f"   Sem sombra: {len(no_shadow_numbers)}")
    
    # Encontrar pares correspondentes
    matching_pairs = shadow_numbers.intersection(no_shadow_numbers)
    shadow_only = shadow_numbers - no_shadow_numbers
    no_shadow_only = no_shadow_numbers - shadow_numbers
    
    print(f"\n🔗 Pares correspondentes encontrados: {len(matching_pairs)}")
    print(f"📋 Números apenas com sombra: {len(shadow_only)}")
    print(f"📋 Números apenas sem sombra: {len(no_shadow_only)}")
    
    if matching_pairs:
        print(f"\n✅ Pares correspondentes disponíveis:")
        matching_list = sorted(list(matching_pairs))
        for i, number in enumerate(matching_list[:10]):  # Mostrar primeiros 10
            print(f"   {i+1:2d}. {number}.jpeg ↔ {number}.jpg")
        
        if len(matching_list) > 10:
            print(f"   ... e mais {len(matching_list) - 10} pares")
    
    if shadow_only:
        print(f"\n⚠️ Apenas com sombra (primeiros 5):")
        for i, number in enumerate(sorted(list(shadow_only))[:5]):
            print(f"   {i+1}. {number}.jpeg")
    
    if no_shadow_only:
        print(f"\n⚠️ Apenas sem sombra (primeiros 5):")
        for i, number in enumerate(sorted(list(no_shadow_only))[:5]):
            print(f"   {i+1}. {number}.jpg")
    
    # Recomendações
    print(f"\n💡 Recomendações:")
    if len(matching_pairs) >= 5:
        print(f"   ✅ Usar {len(matching_pairs)} pares correspondentes para treinamento")
        print(f"   📝 Modificar script para usar apenas pares correspondentes")
    else:
        print(f"   ❌ Poucos pares correspondentes encontrados")
        print(f"   🔍 Verificar se os diretórios estão corretos")
        print(f"   📝 Considerar usar apenas imagens com sombra para treinamento")
    
    return matching_pairs, shadow_only, no_shadow_only


if __name__ == "__main__":
    check_image_pairs()
