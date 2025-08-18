#!/usr/bin/env python3
import argparse
import sys
import os
import yaml
import torch
from torch import nn
from torch.utils.data import DataLoader

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data_manager import ValDataset
from SpA_Former import Generator
from utils import gpu_manage
from eval import test


class SimpleConfig:
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)


def main(config_path):
    with open(config_path, 'r') as f:
        cfg_dict = yaml.safe_load(f)
    config = SimpleConfig(cfg_dict)

    # Desativar CUDA se não disponível
    if not torch.cuda.is_available():
        config.cuda = False

    # Avoid snapshot image assembly inside eval to prevent shape mismatches
    setattr(config, 'snapshot_interval', int(1e9))

    gpu_manage(config)

    val_ds = ValDataset(config)
    val_loader = DataLoader(dataset=val_ds, num_workers=config.threads,
                            batch_size=1, shuffle=False)

    gen = Generator(gpu_ids=config.gpu_ids)
    if getattr(config, 'gen_init', None):
        param = torch.load(config.gen_init, map_location='cpu')
        gen.load_state_dict(param)
    if getattr(config, 'cuda', False) and torch.cuda.is_available():
        gen = gen.cuda()

    criterionMSE = nn.MSELoss()
    if getattr(config, 'cuda', False) and torch.cuda.is_available():
        criterionMSE = criterionMSE.cuda()

    gen.eval()
    with torch.no_grad():
        _ = test(config, val_loader, gen, criterionMSE, epoch=1)

    print("Quick validation completed for", len(val_ds), "images")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./scripts_bkp/config_training_v9_pairs.yml')
    args = parser.parse_args()
    main(args.config)


