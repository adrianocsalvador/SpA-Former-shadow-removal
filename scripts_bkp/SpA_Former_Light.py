import torch
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict
from models.models_utils import weights_init, print_network
from TransFormer import TransformerBlock, OverlapPatchEmbed, Downsample, Upsample


def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1,
                    stride=stride, padding=0, bias=False)

def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
        stride=stride, padding=1, bias=False)


class ResBlock_Light(nn.Module):
    """Versão leve do ResBlock"""
    def __init__(self, in_channel, out_channel):
        super(ResBlock_Light, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        
    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return self.relu(out + residual)


class FFTBlock_Light(nn.Module):
    """Versão leve do FFT Block"""
    def __init__(self, in_channel, out_channel, norm='backward'):
        super(FFTBlock_Light, self).__init__()
        self.conv = nn.Conv2d(in_channel*2, out_channel*2, kernel_size=1, stride=1, bias=False)
        self.relu = nn.ReLU(True)
        self.dim = out_channel
        self.norm = norm
        
    def forward(self, x):
        _, _, H, W = x.shape
        dim = 1
        y = torch.fft.rfft2(x, norm=self.norm)
        y_imag = y.imag
        y_real = y.real
        y_f = torch.cat([y_real, y_imag], dim=dim)
        y = self.relu(self.conv(y_f))
        y_real, y_imag = torch.chunk(y, 2, dim=dim)
        y = torch.complex(y_real, y_imag)
        y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        return y


class Attention_Light(nn.Module):
    """Versão leve do módulo de atenção"""
    def __init__(self, in_channels):
        super(Attention_Light, self).__init__()
        self.out_channels = int(in_channels/2)
        self.conv1 = nn.Conv2d(in_channels, self.out_channels, kernel_size=3, padding=1, stride=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(self.out_channels, 4, kernel_size=1, padding=0, stride=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.sigmoid(self.conv2(out))
        return out


class SAM_Light(nn.Module):
    """Versão leve do SAM"""
    def __init__(self, in_channels, out_channels, attention=1):
        super(SAM_Light, self).__init__()
        self.out_channels = out_channels
        self.conv_in = conv3x3(in_channels, self.out_channels)
        self.relu = nn.ReLU(True)
        
        self.conv1 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(self.out_channels*4, self.out_channels, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(self.out_channels*4, self.out_channels, kernel_size=1, stride=1, padding=0)
        self.relu2 = nn.ReLU(True)
        self.attention = attention
        
        if self.attention:
            self.attention_layer = Attention_Light(in_channels)
            
        self.conv_out = conv1x1(self.out_channels, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        if self.attention:
            weight = self.attention_layer(x)
        
        out = self.conv1(x)
        
        # Simular direções com convoluções simples
        top_up = self.relu(out)
        top_right = self.relu(out)
        top_down = self.relu(out)
        top_left = self.relu(out)
        
        # Aplicar atenção se habilitada
        if self.attention:
            top_up = top_up * weight[:, 0:1, :, :]
            top_right = top_right * weight[:, 1:2, :, :]
            top_down = top_down * weight[:, 2:3, :, :]
            top_left = top_left * weight[:, 3:4, :, :]
        
        out = torch.cat([top_up, top_right, top_down, top_left], dim=1)
        out = self.conv2(out)
        
        # Segunda iteração
        top_up = self.relu(out)
        top_right = self.relu(out)
        top_down = self.relu(out)
        top_left = self.relu(out)
        
        if self.attention:
            top_up = top_up * weight[:, 0:1, :, :]
            top_right = top_right * weight[:, 1:2, :, :]
            top_down = top_down * weight[:, 2:3, :, :]
            top_left = top_left * weight[:, 3:4, :, :]
        
        out = torch.cat([top_up, top_right, top_down, top_left], dim=1)
        out = self.relu2(self.conv3(out))
        mask = self.sigmoid(self.conv_out(out))
        
        return mask


class SpA_Former_Light(nn.Module):
    """Versão leve do SpA-Former otimizada para memória"""
    def __init__(self, inp_channels=3, out_channels=3, dim=32,  # Mantido 32 para compatibilidade
                 num_blocks=[2, 3, 3, 4],  # Reduzido número de blocos
                 num_refinement_blocks=2,  # Reduzido
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2.0,  # Reduzido
                 bias=False,
                 LayerNorm_type='WithBias',
                 dual_pixel_task=False):
        
        super(SpA_Former_Light, self).__init__()
        
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        
        # Encoder com menos blocos
        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], 
                                                              ffn_expansion_factor=ffn_expansion_factor, 
                                                              bias=bias, LayerNorm_type=LayerNorm_type) 
                                             for i in range(num_blocks[0])])
        
        self.down1_2 = Downsample(dim)
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], 
                                                              ffn_expansion_factor=ffn_expansion_factor, 
                                                              bias=bias, LayerNorm_type=LayerNorm_type) 
                                             for i in range(num_blocks[1])])
        
        self.down2_3 = Downsample(int(dim*2**1))
        self.encoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], 
                                                              ffn_expansion_factor=ffn_expansion_factor, 
                                                              bias=bias, LayerNorm_type=LayerNorm_type) 
                                             for i in range(num_blocks[2])])
        
        self.down3_4 = Downsample(int(dim*2**2))
        self.latent = nn.Sequential(*[TransformerBlock(dim=int(dim*2**3), num_heads=heads[3], 
                                                      ffn_expansion_factor=ffn_expansion_factor, 
                                                      bias=bias, LayerNorm_type=LayerNorm_type) 
                                     for i in range(num_blocks[3])])
        
        # Decoder
        self.up4_3 = Upsample(int(dim*2**3))
        self.reduce_chan_level3 = nn.Conv2d(int(dim*2**3), int(dim*2**2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], 
                                                              ffn_expansion_factor=ffn_expansion_factor, 
                                                              bias=bias, LayerNorm_type=LayerNorm_type) 
                                             for i in range(num_blocks[2])])
        
        self.up3_2 = Upsample(int(dim*2**2))
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], 
                                                              ffn_expansion_factor=ffn_expansion_factor, 
                                                              bias=bias, LayerNorm_type=LayerNorm_type) 
                                             for i in range(num_blocks[1])])
        
        self.up2_1 = Upsample(int(dim*2**1))
        self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], 
                                                              ffn_expansion_factor=ffn_expansion_factor, 
                                                              bias=bias, LayerNorm_type=LayerNorm_type) 
                                             for i in range(num_blocks[0])])
        
        # Refinement com menos blocos
        self.refinement = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], 
                                                          ffn_expansion_factor=ffn_expansion_factor, 
                                                          bias=bias, LayerNorm_type=LayerNorm_type) 
                                         for i in range(num_refinement_blocks)])
        
        # Output
        self.output = nn.Conv2d(dim, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        
        # Módulos leves adicionais
        self.SAM1 = SAM_Light(dim, dim, 1)
        
        # ResBlocks leves (menos blocos)
        self.res_block1 = ResBlock_Light(dim, dim)
        self.res_block2 = ResBlock_Light(dim, dim)
        self.res_block3 = ResBlock_Light(dim, dim)
        self.res_block4 = ResBlock_Light(dim, dim)
        self.res_block5 = ResBlock_Light(dim, dim)
        
        # FFT Blocks leves (menos blocos)
        self.fft_block1 = FFTBlock_Light(dim, dim)
        self.fft_block2 = FFTBlock_Light(dim, dim)
        self.fft_block3 = FFTBlock_Light(dim, dim)
        self.fft_block4 = FFTBlock_Light(dim, dim)
        self.fft_block5 = FFTBlock_Light(dim, dim)
        
    def forward(self, x):
        # Encoder
        inp_enc_level1 = self.patch_embed(x)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        
        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)
        
        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)
        
        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(inp_enc_level4)
        
        # Decoder
        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)
        
        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)
        
        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        
        out_dec_level1 = self.refinement(out_dec_level1)
        
        # Processamento adicional leve
        out = out_dec_level1
        
        # ResBlocks leves
        out = F.relu(self.res_block1(out) + out + self.fft_block1(out))
        out = F.relu(self.res_block2(out) + out + self.fft_block2(out))
        out = F.relu(self.res_block3(out) + out + self.fft_block3(out))
        
        # Atenção leve
        Attention1 = self.SAM1(out)
        out = F.relu(self.res_block4(out) * Attention1 + out + self.fft_block4(out))
        out = F.relu(self.res_block5(out) * Attention1 + out + self.fft_block5(out))
        
        # Output
        out = self.output(out)
        
        return Attention1, out


class Generator_Light(nn.Module):
    """Gerador leve"""
    def __init__(self, gpu_ids):
        super().__init__()
        self.gpu_ids = gpu_ids
        self.gen = nn.Sequential(OrderedDict([('gen', SpA_Former_Light())]))
        self.gen.apply(weights_init)

    def forward(self, x):
        if self.gpu_ids:
            return nn.parallel.data_parallel(self.gen, x, self.gpu_ids)
        else:
            return self.gen(x)
