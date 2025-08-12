import os
import random
import shutil
import yaml
import time
import argparse

import torch
from torch import nn
from torch.backends import cudnn
from torch import optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn import functional as F

# Adicionar o diretório raiz ao path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_manager import TrainDataset,ValDataset
from SpA_Former import Generator
from models.dis.dis import Discriminator
import utils
from utils import gpu_manage, save_image, checkpoint
from eval import test
from log_report import LogReport
from log_report import TestReport

class Config:
    """Classe para substituir AttrMap"""
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)

def train(config):
    gpu_manage(config)

    ### DATASET LOAD ###
    print('===> Loading datasets')
    
    train_dataset = TrainDataset(config)
    validation_dataset = ValDataset(config)
    print('train dataset:', len(train_dataset))
    print('validation dataset:', len(validation_dataset))

    training_data_loader = DataLoader(dataset=train_dataset, num_workers=config.threads, batch_size=config.batchsize, shuffle=True)
    validation_data_loader = DataLoader(dataset=validation_dataset, num_workers=config.threads, batch_size=config.validation_batchsize, shuffle=False)
    
    
    ### MODELS LOAD ###
    print('===> Loading models')

    gen = Generator(gpu_ids=config.gpu_ids)

    if hasattr(config, 'gen_init') and config.gen_init is not None:
        param = torch.load(config.gen_init)
        gen.load_state_dict(param)
        print('load {} as pretrained model'.format(config.gen_init))

    dis = Discriminator(in_ch=config.in_ch, out_ch=config.out_ch, gpu_ids=config.gpu_ids)

    if hasattr(config, 'dis_init') and config.dis_init is not None:
        param = torch.load(config.dis_init)
        dis.load_state_dict(param)
        print('load {} as pretrained model'.format(config.dis_init))

    # setup optimizer
    opt_gen = optim.Adam(gen.parameters(), lr=config.lr, betas=(config.beta1, 0.999), weight_decay=0.00001)
    opt_dis = optim.Adam(dis.parameters(), lr=config.lr, betas=(config.beta1, 0.999), weight_decay=0.00001)

    real_a = torch.FloatTensor(config.batchsize, config.in_ch, config.width, config.height)
    real_b = torch.FloatTensor(config.batchsize, config.out_ch, config.width, config.height)
    M = torch.FloatTensor(config.batchsize, config.width, config.height)

    criterionL1 = nn.L1Loss()
    criterionMSE = nn.MSELoss()
    criterionSoftplus = nn.Softplus()

    if config.cuda:
        gen = gen.cuda()
        dis = dis.cuda()
        criterionL1 = criterionL1.cuda()
        criterionMSE = criterionMSE.cuda()
        criterionSoftplus = criterionSoftplus.cuda()
        real_a = real_a.cuda()
        real_b = real_b.cuda()
        M = M.cuda()

    real_a = Variable(real_a)
    real_b = Variable(real_b)

    logreport = LogReport(log_dir=config.out_dir)
    validationreport = TestReport(log_dir=config.out_dir)

    print('===> begin')
    start_time=time.time()
    # main
    for epoch in range(1, config.epoch + 1):
        epoch_start_time = time.time()
        for iteration, batch in enumerate(training_data_loader, 1):
            real_a_cpu, real_b_cpu, M_cpu = batch[0], batch[1], batch[2]
            real_a.resize_(real_a_cpu.size()).copy_(real_a_cpu)
            real_b.resize_(real_b_cpu.size()).copy_(real_b_cpu)
            M.resize_(M_cpu.size()).copy_(M_cpu)
            att, fake_b = gen.forward(real_a)

            ################
            ### Update D ###
            ################
            opt_dis.zero_grad()

            # train with fake
            fake_ab = torch.cat((real_a, fake_b), 1)
            pred_fake = dis.forward(fake_ab.detach())
            batch_size = pred_fake.size()[0]
            label_fake = torch.zeros(batch_size, 1, pred_fake.size()[2], pred_fake.size()[3])
            if config.cuda:
                label_fake = label_fake.cuda()
            label_fake = Variable(label_fake)
            loss_d_fake = criterionSoftplus(pred_fake).mean()

            # train with real
            real_ab = torch.cat((real_a, real_b), 1)
            pred_real = dis.forward(real_ab)
            label_real = torch.ones(batch_size, 1, pred_real.size()[2], pred_real.size()[3])
            if config.cuda:
                label_real = label_real.cuda()
            label_real = Variable(label_real)
            loss_d_real = criterionSoftplus(-pred_real).mean()

            # Combined loss
            loss_d = (loss_d_real + loss_d_fake) * 0.5
            loss_d.backward()
            opt_dis.step()

            ################
            ### Update G ###
            ################
            opt_gen.zero_grad()

            # First, G(A) should fake the discriminator
            fake_ab = torch.cat((real_a, fake_b), 1)
            pred_fake = dis.forward(fake_ab)
            loss_g_gan = criterionSoftplus(-pred_fake).mean()

            # Second, G(A) = B
            loss_g_l1 = criterionL1(fake_b, real_b) * config.lamb
            loss_g = loss_g_gan + loss_g_l1

            loss_g.backward()
            opt_gen.step()

            # determine approximate time left
            epoch_end_time = time.time()
            epoch_time = epoch_end_time - epoch_start_time

            if iteration % 100 == 0:
                print("===> Epoch[{}]({}/{}): Loss_D: {:.4f} Loss_G: {:.4f} Loss_G_L1: {:.4f} Time: {:.2f}s".format(
                    epoch, iteration, len(training_data_loader), loss_d.item(), loss_g.item(), loss_g_l1.item(), epoch_time))

            if iteration % 100 == 0:
                logreport.log({'epoch': epoch, 'iteration': iteration, 'Loss_D': loss_d.item(), 'Loss_G': loss_g.item(), 'Loss_G_L1': loss_g_l1.item()})

        # validation
        if epoch % config.snapshot_interval == 0:
            print("===> Validating")
            validationreport.clear()
            for iteration, batch in enumerate(validation_data_loader, 1):
                real_a_cpu, real_b_cpu, M_cpu = batch[0], batch[1], batch[2]
                real_a.resize_(real_a_cpu.size()).copy_(real_a_cpu)
                real_b.resize_(real_b_cpu.size()).copy_(real_b_cpu)
                M.resize_(M_cpu.size()).copy_(M_cpu)
                att, fake_b = gen.forward(real_a)
                validationreport.log({'epoch': epoch, 'iteration': iteration, 'Loss_G_L1': criterionL1(fake_b, real_b).item()})

            validationreport.log({'epoch': epoch, 'avg_loss': validationreport.avg_loss()})
            print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, validationreport.avg_loss()))

        # save checkpoints
        if epoch % config.snapshot_interval == 0:
            checkpoint(gen, epoch, config.out_dir, 'gen')
            checkpoint(dis, epoch, config.out_dir, 'dis')

        # test
        if epoch % config.snapshot_interval == 0:
            test(config, gen, validation_data_loader, validationreport, epoch)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SpA-Former')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()

    with open(args.config, 'r', encoding='UTF-8') as f:
        config_dict = yaml.safe_load(f)
    
    config = Config(config_dict)
    
    print("🚀 Iniciando treinamento com configuração:")
    print(f"📂 Dataset: {config.datasets_dir}")
    print(f"📂 Validação: {config.valset_dir}")
    print(f"📊 Épocas: {config.epoch}")
    print(f"📊 Batch size: {config.batchsize}")
    print(f"📊 Imagens: {config.n_data}")
    print(f"💾 Saída: {config.out_dir}")
    
    train(config) 