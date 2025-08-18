import numpy as np
import torch
#from skimage.measure import compare_ssim as SSIM
from skimage.metrics import structural_similarity as SSIM

from torch.autograd import Variable

from utils import save_image

from sklearn import metrics
import math
#from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import mean_squared_error as compare_mse
from skimage import color
import cv2



def test(config, test_data_loader, gen, criterionMSE, epoch):
    avg_mae = 0
    avg_psnr = 0
    avg_ssim = 0
    mae = 0


    
    with torch.no_grad():
        for i, batch in enumerate(test_data_loader):
            x, t = Variable(batch[0]), Variable(batch[1])
            if config.cuda:
                x = x.cuda(0)
                t = t.cuda(0)

            att, out = gen(x)
       
        if epoch % config.snapshot_interval == 0:
            h = 1
            w = 3
            c = 3
            # Derivar dimensões a partir da saída (N, C, H, W)
            p = out.shape[2]
            q = out.shape[3]

            allim = np.zeros((h, w, c, p, q))
            x_ = x.detach().cpu().numpy()[0]
            t_ = t.detach().cpu().numpy()[0]
            out_ = out.detach().cpu().numpy()[0]
            in_rgb = x_[:3]
            t_rgb = t_[:3]
            out_rgb = np.clip(out_[:3], 0, 1)
            allim[0, 0, :] = in_rgb * 255
            allim[0, 1, :] = out_rgb * 255
            allim[0, 2, :] = t_rgb * 255
            
            allim = allim.transpose(0, 3, 1, 4, 2)
            allim = allim.reshape((h*p, w*q, c))

            #save_image(config.out_dir, allim, i, epoch)
    

        mse = criterionMSE(out, t)
    
        psnr = 10 * np.log10(1 / mse.item())
       

        img1 = np.tensordot(out.cpu().numpy()[0, :3].transpose(1, 2, 0), [0.298912, 0.586611, 0.114478], axes=1)
        img2 = np.tensordot(t.cpu().numpy()[0, :3].transpose(1, 2, 0), [0.298912, 0.586611, 0.114478], axes=1)
        
        
        '''
        out = out.cpu().numpy()
        t = t.cpu().numpy()
        out_lab = cv2.cvtColor(out,cv2.COLOR_RGB2LAB)
        t_lab = cv2.cvtColor(t,cv2.COLOR_RGB2LAB)
 
        
        mae = np.mean(abs(out_lab - t_lab))
        '''
    
        ssim = SSIM(img1, img2, data_range=1.0)
        avg_mae += mae
        avg_psnr += psnr
        avg_ssim += ssim


        
    avg_mae = avg_mae / len(test_data_loader)
    avg_psnr = avg_psnr / len(test_data_loader)
    avg_ssim = avg_ssim / len(test_data_loader)

    

    print("===> Avg. MAE: {:.4f}".format(avg_mae))
    print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr))
    print("===> Avg. SSIM: {:.4f} dB".format(avg_ssim))


    
    log_test = {}
    log_test['epoch'] = epoch
    log_test['mae'] = avg_mae
    log_test['psnr'] = avg_psnr
    log_test['ssim'] = avg_ssim


    
    
    

    return log_test


def _generate_starts(total: int, window: int, stride: int):
    starts = list(range(0, max(total - window + 1, 1), stride))
    last_start = max(total - window, 0)
    if not starts or starts[-1] != last_start:
        starts.append(last_start)
    return starts


def test_sliding(config, test_data_loader, gen, criterionMSE, epoch):
    avg_mae = 0
    avg_psnr = 0
    avg_ssim = 0
    mae = 0

    window_w = getattr(config, 'window_width', 256)
    window_h = getattr(config, 'window_height', 256)
    stride_w = getattr(config, 'stride_width', 128)
    stride_h = getattr(config, 'stride_height', 128)

    gen.eval()
    with torch.no_grad():
        for i, batch in enumerate(test_data_loader):
            x, t = batch[0], batch[1]
            x = torch.as_tensor(x).unsqueeze(0) if x.ndim == 3 else torch.as_tensor(x)
            t = torch.as_tensor(t).unsqueeze(0) if t.ndim == 3 else torch.as_tensor(t)

            # Validar no dispositivo atual conforme config
            if getattr(config, 'cuda', False) and torch.cuda.is_available():
                x = x.cuda()
                t = t.cuda()

            _, C, H, W = x.shape
            out_sum = torch.zeros((1, C, H, W), device=x.device)
            out_cnt = torch.zeros((1, 1, H, W), device=x.device)

            starts_y = _generate_starts(H, window_h, stride_h)
            starts_x = _generate_starts(W, window_w, stride_w)

            for y in starts_y:
                for x0 in starts_x:
                    x_patch = x[:, :, y:y+window_h, x0:x0+window_w]
                    att, out_patch = gen(x_patch)
                    out_sum[:, :, y:y+window_h, x0:x0+window_w] += out_patch
                    out_cnt[:, :, y:y+window_h, x0:x0+window_w] += 1

            out = out_sum / torch.clamp(out_cnt, min=1)

            mse = criterionMSE(out, t)
            psnr = 10 * np.log10(1 / mse.item())

            img1 = torch.tensordot(out[0, :3].permute(1, 2, 0), torch.tensor([0.298912, 0.586611, 0.114478], device=out.device, dtype=out.dtype), dims=1).detach().cpu().numpy()
            img2 = torch.tensordot(t[0, :3].permute(1, 2, 0), torch.tensor([0.298912, 0.586611, 0.114478], device=t.device, dtype=t.dtype), dims=1).detach().cpu().numpy()

            ssim = SSIM(img1, img2, data_range=1.0)
            avg_mae += mae
            avg_psnr += psnr
            avg_ssim += ssim

    avg_mae = avg_mae / len(test_data_loader)
    avg_psnr = avg_psnr / len(test_data_loader)
    avg_ssim = avg_ssim / len(test_data_loader)

    print("===> [Sliding] Avg. MAE: {:.4f}".format(avg_mae))
    print("===> [Sliding] Avg. PSNR: {:.4f} dB".format(avg_psnr))
    print("===> [Sliding] Avg. SSIM: {:.4f} dB".format(avg_ssim))

    log_test = {
        'epoch': epoch,
        'mae': avg_mae,
        'psnr': avg_psnr,
        'ssim': avg_ssim,
        'mode': 'sliding'
    }
    return log_test
