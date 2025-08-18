#!/usr/bin/env python3
"""
Cria um subset do dataset no formato de pares (A/C) copiando N imagens para treino
e M para teste, preservando a estrutura:

dst_root/
  train/
    train_A/
    train_C/
    train_list.txt
  test/
    test_A/
    test_C/
    val_list.txt

Uso:
  python scripts_bkp/create_subset_pairs.py \
    --src /mnt/.../dataset_v9_pairs \
    --dst /mnt/.../dataset_v9_pairs_small_100_20 \
    --train-count 100 --test-count 20
"""

import os
import shutil
import argparse
from pathlib import Path


def ensure_pairs(src_dir_a: str, src_dir_c: str, ext: str):
    files_a = sorted([f for f in os.listdir(src_dir_a) if f.endswith(ext)])
    files_c = set([f for f in os.listdir(src_dir_c) if f.endswith(ext)])
    paired = [f for f in files_a if f in files_c]
    return paired


def copy_subset(src_root: str, dst_root: str, train_count: int, test_count: int, ext: str = '.png'):
    # Origens
    src_train_a = os.path.join(src_root, 'train', 'train_A')
    src_train_c = os.path.join(src_root, 'train', 'train_C')
    src_test_a = os.path.join(src_root, 'test', 'test_A')
    src_test_c = os.path.join(src_root, 'test', 'test_C')

    # Destinos
    dst_train_a = os.path.join(dst_root, 'train', 'train_A')
    dst_train_c = os.path.join(dst_root, 'train', 'train_C')
    dst_test_a = os.path.join(dst_root, 'test', 'test_A')
    dst_test_c = os.path.join(dst_root, 'test', 'test_C')

    for d in [dst_train_a, dst_train_c, dst_test_a, dst_test_c]:
        os.makedirs(d, exist_ok=True)

    # Listas pareadas
    train_paired = ensure_pairs(src_train_a, src_train_c, ext)[:train_count]
    test_paired = ensure_pairs(src_test_a, src_test_c, ext)[:test_count]

    if len(train_paired) < train_count:
        print(f"Aviso: apenas {len(train_paired)} pares de treino encontrados com ext {ext}")
    if len(test_paired) < test_count:
        print(f"Aviso: apenas {len(test_paired)} pares de teste encontrados com ext {ext}")

    # Copiar treino
    for fname in train_paired:
        shutil.copy2(os.path.join(src_train_a, fname), os.path.join(dst_train_a, fname))
        shutil.copy2(os.path.join(src_train_c, fname), os.path.join(dst_train_c, fname))

    # Copiar teste
    for fname in test_paired:
        shutil.copy2(os.path.join(src_test_a, fname), os.path.join(dst_test_a, fname))
        shutil.copy2(os.path.join(src_test_c, fname), os.path.join(dst_test_c, fname))

    # Gerar listas
    train_list_path = os.path.join(dst_root, 'train', 'train_list.txt')
    val_list_path = os.path.join(dst_root, 'test', 'val_list.txt')
    with open(train_list_path, 'w') as f:
        for fname in sorted(os.listdir(dst_train_a)):
            if fname.endswith(ext):
                f.write(f"{fname}\n")
    with open(val_list_path, 'w') as f:
        for fname in sorted(os.listdir(dst_test_a)):
            if fname.endswith(ext):
                f.write(f"{fname}\n")

    print("\n✅ Subset criado com sucesso!")
    print(f"📁 Destino: {dst_root}")
    print(f"  Treino: {len(os.listdir(dst_train_a))} A / {len(os.listdir(dst_train_c))} C")
    print(f"  Teste : {len(os.listdir(dst_test_a))} A / {len(os.listdir(dst_test_c))} C")
    print(f"  Listas: {train_list_path}, {val_list_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', required=True, help='Raiz do dataset original (contém train/ e test/)')
    parser.add_argument('--dst', required=True, help='Raiz do dataset destino (subset)')
    parser.add_argument('--train-count', type=int, default=100)
    parser.add_argument('--test-count', type=int, default=20)
    parser.add_argument('--ext', default='.png')
    args = parser.parse_args()

    copy_subset(args.src, args.dst, args.train_count, args.test_count, args.ext)


if __name__ == '__main__':
    main()


