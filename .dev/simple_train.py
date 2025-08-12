#!/usr/bin/env python3

import os
import sys
import yaml
import argparse
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import time

# Adicionar o diretório raiz ao path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from SpA_Former import Generator
from models.dis.dis import Discriminator
import utils
from utils import gpu_manage, save_image

class SimpleConfig:
    """Classe simples para substituir AttrMap"""
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)

def simple_checkpoint(model, epoch, out_dir, name):
    """Função simplificada para salvar checkpoint"""
    model_dir = os.path.join(out_dir, 'models')
    os.makedirs(model_dir, exist_ok=True)
    model_out_path = os.path.join(model_dir, f'{name}_model_epoch_{epoch}.pth')
    torch.save(model.state_dict(), model_out_path)
    print(f"💾 {name} salvo: {model_out_path}")

def simple_train(config):
    """Treinamento simplificado do SpA-Former"""
    
    print("🚀 Iniciando treinamento simplificado...")
    
    # Configurar GPU
    if config.cuda:
        torch.cuda.set_device(config.gpu_ids[0])
    
    # Carregar modelos
    print("📦 Carregando modelos...")
    gen = Generator(gpu_ids=config.gpu_ids)
    dis = Discriminator(in_ch=config.in_ch, out_ch=config.out_ch, gpu_ids=config.gpu_ids)
    
    if config.cuda:
        gen = gen.cuda()
        dis = dis.cuda()
    
    # Otimizadores
    opt_gen = optim.Adam(gen.parameters(), lr=config.lr, betas=(config.beta1, 0.999))
    opt_dis = optim.Adam(dis.parameters(), lr=config.lr, betas=(config.beta1, 0.999))
    
    # Critérios
    criterionL1 = nn.L1Loss()
    criterionSoftplus = nn.Softplus()
    
    if config.cuda:
        criterionL1 = criterionL1.cuda()
        criterionSoftplus = criterionSoftplus.cuda()
    
    # Criar diretório de saída
    os.makedirs(config.out_dir, exist_ok=True)
    
    print("🎯 Iniciando loop de treinamento...")
    print(f"📊 Épocas: {config.epoch}")
    print(f"📊 Batch size: {config.batchsize}")
    print(f"📊 Imagens: {config.n_data}")
    
    # Loop de treinamento simplificado
    for epoch in range(1, config.epoch + 1):
        epoch_start_time = time.time()
        
        print(f"🔄 Época {epoch}/{config.epoch}")
        
        # Simular algumas iterações (sem dataset real por enquanto)
        for iteration in range(1, 11):  # 10 iterações por época para teste
            # Criar dados simulados
            real_a = torch.randn(config.batchsize, config.in_ch, config.width, config.height)
            real_b = torch.randn(config.batchsize, config.out_ch, config.width, config.height)
            
            if config.cuda:
                real_a = real_a.cuda()
                real_b = real_b.cuda()
            
            real_a = Variable(real_a)
            real_b = Variable(real_b)
            
            # Forward pass do gerador
            att, fake_b = gen.forward(real_a)
            
            # Treinar discriminador
            opt_dis.zero_grad()
            
            fake_ab = torch.cat((real_a, fake_b), 1)
            pred_fake = dis.forward(fake_ab.detach())
            loss_d_fake = criterionSoftplus(pred_fake).mean()
            
            real_ab = torch.cat((real_a, real_b), 1)
            pred_real = dis.forward(real_ab)
            loss_d_real = criterionSoftplus(-pred_real).mean()
            
            loss_d = (loss_d_real + loss_d_fake) * 0.5
            loss_d.backward()
            opt_dis.step()
            
            # Treinar gerador
            opt_gen.zero_grad()
            
            fake_ab = torch.cat((real_a, fake_b), 1)
            pred_fake = dis.forward(fake_ab)
            loss_g_gan = criterionSoftplus(-pred_fake).mean()
            
            loss_g_l1 = criterionL1(fake_b, real_b) * config.lamb
            loss_g = loss_g_gan + loss_g_l1
            
            loss_g.backward()
            opt_gen.step()
            
            if iteration % 5 == 0:
                print(f"  Iteração {iteration}: Loss_D={loss_d.item():.4f}, Loss_G={loss_g.item():.4f}")
        
        # Salvar checkpoint
        if epoch % config.snapshot_interval == 0:
            simple_checkpoint(gen, epoch, config.out_dir, 'gen')
            simple_checkpoint(dis, epoch, config.out_dir, 'dis')
            print(f"💾 Checkpoint salvo na época {epoch}")
        
        epoch_time = time.time() - epoch_start_time
        print(f"✅ Época {epoch} concluída em {epoch_time:.2f}s")
    
    print("🎉 Treinamento concluído!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Treinamento simplificado do SpA-Former')
    parser.add_argument('--config', type=str, required=True, help='Caminho para o arquivo de configuração')
    args = parser.parse_args()
    
    with open(args.config, 'r', encoding='UTF-8') as f:
        config_dict = yaml.safe_load(f)
    
    config = SimpleConfig(config_dict)
    
    print("🚀 Treinamento simplificado do SpA-Former")
    print(f"📂 Configuração: {args.config}")
    print(f"📊 Dataset: {config.datasets_dir}")
    print(f"📊 Validação: {config.valset_dir}")
    print(f"📊 Épocas: {config.epoch}")
    print(f"📊 Batch size: {config.batchsize}")
    print(f"📊 Imagens: {config.n_data}")
    print(f"💾 Saída: {config.out_dir}")
    
    simple_train(config) 