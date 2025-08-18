#!/usr/bin/env python3
"""
Script de teste simples para identificar onde está o problema no treinamento
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

# Configurar memória ULTRA
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:128'

def clear_memory():
    """Limpa memória GPU"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    import gc
    gc.collect()

class SimpleDataset:
    """Dataset simples para teste"""
    
    def __init__(self, num_samples=10):
        self.num_samples = num_samples
        # Criar dados dummy
        self.data = []
        for i in range(num_samples):
            # Criar imagens dummy 3x512x512
            shadow = torch.rand(3, 512, 512)
            shadow_free = torch.rand(3, 512, 512)
            self.data.append((shadow, shadow_free))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def test_simple_training():
    """Teste simples de treinamento"""
    print("🧪 Teste simples de treinamento...")
    
    # Verificar CUDA
    if torch.cuda.is_available():
        print(f"✅ CUDA disponível: {torch.cuda.get_device_name(0)}")
        print(f"📊 Memória GPU: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("❌ CUDA não disponível")
        return
    
    # Criar dataset simples
    print("\n📊 Criando dataset simples...")
    dataset = SimpleDataset(num_samples=5)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    
    print(f"✅ Dataset criado com {len(dataset)} amostras")
    
    # Testar carregamento de dados
    print("\n🧪 Testando carregamento de dados...")
    
    for batch_idx, (shadow_batch, shadow_free_batch) in enumerate(dataloader):
        print(f"\n📦 Batch {batch_idx}:")
        print(f"   ✅ Shadow batch: {shadow_batch.shape}, dtype: {shadow_batch.dtype}")
        print(f"   ✅ Shadow-free batch: {shadow_free_batch.shape}, dtype: {shadow_free_batch.dtype}")
        print(f"   📊 Shadow batch min/max: {shadow_batch.min():.3f}/{shadow_batch.max():.3f}")
        print(f"   📊 Shadow-free batch min/max: {shadow_free_batch.min():.3f}/{shadow_free_batch.max():.3f}")
        
        # Mover para GPU
        shadow_batch = shadow_batch.cuda()
        shadow_free_batch = shadow_free_batch.cuda()
        
        print(f"   ✅ Movido para GPU: {shadow_batch.device}")
        
        # Testar apenas o primeiro batch
        if batch_idx == 0:
            break
    
    # Testar modelos simples
    print("\n🧪 Testando modelos simples...")
    
    try:
        # Importar SpA-Former
        sys.path.append('.')
        from SpA_Former import SpA_former
        from models.dis.dis import Discriminator
        
        print("✅ Modelos importados com sucesso")
        
        # Criar modelos
        generator = SpA_former()
        discriminator = Discriminator(in_ch=3, out_ch=3, gpu_ids=[0])
        
        print("✅ Modelos criados com sucesso")
        
        # Mover para GPU
        generator = generator.cuda()
        discriminator = discriminator.cuda()
        
        print("✅ Modelos movidos para GPU")
        
        # Testar forward pass
        print("\n🧪 Testando forward pass...")
        
        # Criar dados de teste
        test_input = torch.rand(1, 3, 512, 512).cuda()
        
        print(f"   📊 Input shape: {test_input.shape}")
        print(f"   📊 Input device: {test_input.device}")
        print(f"   📊 Input min/max: {test_input.min():.3f}/{test_input.max():.3f}")
        
        # Testar gerador
        try:
            with torch.no_grad():
                output = generator(test_input)
            print(f"   ✅ Generator output: {output.shape}")
            print(f"   📊 Generator output min/max: {output.min():.3f}/{output.max():.3f}")
        except Exception as e:
            print(f"   ❌ Erro no gerador: {e}")
        
        # Testar discriminador
        try:
            with torch.no_grad():
                disc_output = discriminator(test_input)
            print(f"   ✅ Discriminator output: {disc_output.shape}")
            print(f"   📊 Discriminator output min/max: {disc_output.min():.3f}/{disc_output.max():.3f}")
        except Exception as e:
            print(f"   ❌ Erro no discriminador: {e}")
        
        # Testar treinamento simples
        print("\n🧪 Testando treinamento simples...")
        
        # Criar otimizadores
        optimizer_g = torch.optim.Adam(generator.parameters(), lr=0.0001)
        optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=0.0001)
        
        # Criar loss functions
        criterion_gan = nn.BCELoss().cuda()
        criterion_l1 = nn.L1Loss().cuda()
        
        # Testar um passo de treinamento
        for batch_idx, (shadow_batch, shadow_free_batch) in enumerate(dataloader):
            print(f"\n🔄 Passo de treinamento {batch_idx}:")
            
            # Mover para GPU
            shadow_batch = shadow_batch.cuda()
            shadow_free_batch = shadow_free_batch.cuda()
            
            print(f"   📊 Shadow batch: {shadow_batch.shape}")
            print(f"   📊 Shadow-free batch: {shadow_free_batch.shape}")
            
            # Forward pass do gerador
            try:
                fake_output = generator(shadow_batch)
                print(f"   ✅ Generator forward pass: {fake_output.shape}")
                
                # Calcular loss do gerador
                fake_labels = torch.ones(fake_output.size(0), 1).cuda()
                fake_output_flat = fake_output.view(fake_output.size(0), -1)
                fake_output_flat = torch.sigmoid(fake_output_flat.mean(dim=1, keepdim=True))
                loss_g_gan = criterion_gan(fake_output_flat, fake_labels)
                loss_g_l1 = criterion_l1(fake_output, shadow_free_batch)
                loss_g = loss_g_gan + 10 * loss_g_l1
                
                print(f"   📊 Generator loss: {loss_g.item():.6f}")
                
                # Backward pass do gerador
                optimizer_g.zero_grad()
                loss_g.backward()
                optimizer_g.step()
                
                print(f"   ✅ Generator backward pass concluído")
                
            except Exception as e:
                print(f"   ❌ Erro no gerador: {e}")
                import traceback
                traceback.print_exc()
            
            # Forward pass do discriminador
            try:
                real_output = discriminator(shadow_free_batch)
                fake_output = generator(shadow_batch).detach()
                fake_output_d = discriminator(fake_output)
                
                print(f"   ✅ Discriminator forward pass: {real_output.shape}")
                
                # Calcular loss do discriminador
                real_labels = torch.ones(real_output.size(0), 1).cuda()
                fake_labels = torch.zeros(fake_output_d.size(0), 1).cuda()
                
                real_output_flat = real_output.view(real_output.size(0), -1)
                fake_output_flat = fake_output_d.view(fake_output_d.size(0), -1)
                real_output_flat = torch.sigmoid(real_output_flat.mean(dim=1, keepdim=True))
                fake_output_flat = torch.sigmoid(fake_output_flat.mean(dim=1, keepdim=True))
                
                loss_d_real = criterion_gan(real_output_flat, real_labels)
                loss_d_fake = criterion_gan(fake_output_flat, fake_labels)
                loss_d = (loss_d_real + loss_d_fake) * 0.5
                
                print(f"   📊 Discriminator loss: {loss_d.item():.6f}")
                
                # Backward pass do discriminador
                optimizer_d.zero_grad()
                loss_d.backward()
                optimizer_d.step()
                
                print(f"   ✅ Discriminator backward pass concluído")
                
            except Exception as e:
                print(f"   ❌ Erro no discriminador: {e}")
                import traceback
                traceback.print_exc()
            
            # Testar apenas o primeiro batch
            if batch_idx == 0:
                break
        
        print("\n✅ Teste de treinamento concluído com sucesso!")
        
    except Exception as e:
        print(f"❌ Erro geral: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_simple_training()
