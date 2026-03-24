
import time
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skvideo.measure import niqe
import torch.nn.functional as F

import torch
import torch.nn.functional as F
import math

def masked_psnr(pred, target, mask, eps=1e-8):
    """
    pred, target: [B,3,H,W] in [0,1]
    mask: [B,1,H,W] with 1 = include, 0 = ignore
    Returns: psnr(float), N_effective(int)
    """
    # expand mask to channels
    m = mask.expand_as(pred).to(pred.dtype)
    sse = ((pred - target) ** 2 * m).sum()              # sum of squared error over masked pixels
    N  = m.sum()                                        # number of masked channel-pixels
    N = N.clamp_min(1.0)
    mse = sse / N
    psnr = 10.0 * torch.log10(1.0 / (mse + eps))
    return psnr.item(), int(N.item())


def gaussian_window(channels, kernel_size=11, sigma=1.5, device='cpu'):
    import math
    k = torch.arange(kernel_size, device=device) - (kernel_size - 1)/2
    g = torch.exp(-(k**2)/(2*sigma*sigma))
    g = (g / g.sum()).view(1,1,1,-1)
    window = g.transpose(-1,-2) @ g  # 2D
    window = window.expand(channels, 1, kernel_size, kernel_size).contiguous()
    return window

def masked_ssim(pred, target, mask, data_range=1.0, kernel_size=11, sigma=1.5, eps=1e-8):
    """
    pred, target: [B,3,H,W] in [0,1]
    mask: [B,1,H,W] with 1 = include, 0 = ignore
    Returns: weighted SSIM over masked area, N_effective pixels (not used for weighting SSIM usually)
    """
    C = pred.size(1)
    window = gaussian_window(C, kernel_size, sigma, device=pred.device)
    padding = kernel_size // 2

    # per-channel stats
    mu_x = F.conv2d(pred, window, groups=C, padding=padding)
    mu_y = F.conv2d(target, window, groups=C, padding=padding)

    mu_x2  = mu_x * mu_x
    mu_y2  = mu_y * mu_y
    mu_xy  = mu_x * mu_y

    sigma_x2 = F.conv2d(pred * pred, window, groups=C, padding=padding) - mu_x2
    sigma_y2 = F.conv2d(target * target, window, groups=C, padding=padding) - mu_y2
    sigma_xy = F.conv2d(pred * target, window, groups=C, padding=padding) - mu_xy

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    ssim_map = ((2*mu_xy + C1) * (2*sigma_xy + C2)) / ((mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2) + eps)
    # average over channels → [B,1,H,W]
    ssim_map = ssim_map.mean(dim=1, keepdim=True)

    # weight by mask; optionally shrink mask with same window to avoid border bias
    weight = F.conv2d(mask, window[:1], padding=padding)  # smooth mask
    ssim_weighted = (ssim_map * weight).sum()
    Wsum = weight.sum().clamp_min(1.0)
    return (ssim_weighted / Wsum).item(), int(mask.sum().item())

class AverageMeter():
    """ Computes and stores the average and current value """

    def __init__(self):
        self.reset()

    def reset(self):
        """ Reset all statistics """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """ Update statistics """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """ Computes the precision@k for the specified values of k """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    # one-hot case
    if target.ndimension() > 1:
        target = target.max(1)[1]

    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(1.0 / batch_size))

    return res


def compute_psnr_ssim(recoverd, clean):
    assert recoverd.shape == clean.shape
    recoverd = np.clip(recoverd.detach().cpu().numpy(), 0, 1)
    clean = np.clip(clean.detach().cpu().numpy(), 0, 1)

    recoverd = recoverd.transpose(0, 2, 3, 1)
    clean = clean.transpose(0, 2, 3, 1)
    psnr = 0
    ssim = 0


    for i in range(recoverd.shape[0]):
        # psnr_val += compare_psnr(clean[i], recoverd[i])
        # ssim += compare_ssim(clean[i], recoverd[i], multichannel=True)
        psnr += peak_signal_noise_ratio(clean[i], recoverd[i], data_range=1)
        ssim += structural_similarity(clean[i], recoverd[i], data_range=1, multichannel=True, channel_axis=2)

    return psnr / recoverd.shape[0], ssim / recoverd.shape[0], recoverd.shape[0]


def compute_psnr_ssim_allcolorspaces(recoverd, clean):
    assert recoverd.shape == clean.shape
    recoverd = np.clip(recoverd.detach().cpu().numpy(), 0, 1)
    clean = np.clip(clean.detach().cpu().numpy(), 0, 1)

    recoverd = recoverd.transpose(0, 2, 3, 1)
    clean = clean.transpose(0, 2, 3, 1)
    psnr_rgb = 0
    ssim_rgb = 0
    psnr_hsv = 0
    ssim_hsv = 0
    psnr_ycbcr = 0
    ssim_ycbcr = 0
    psnr_lab = 0
    ssim_lab = 0
    import cv2

    for i in range(recoverd.shape[0]):
        # psnr_val += compare_psnr(clean[i], recoverd[i])
        # ssim += compare_ssim(clean[i], recoverd[i], multichannel=True)
        clean_img = (clean[i] * 255).astype(np.uint8)
        recovered_img = (recoverd[i] * 255).astype(np.uint8)
        # clean_img_ycbcr = cv2.cvtColor(clean_img, cv2.COLOR_RGB2GRAY)
        # recovered_img_ycbcr = cv2.cvtColor(recovered_img, cv2.COLOR_RGB2GRAY)
        clean_img_ycbcr = cv2.cvtColor(clean_img, cv2.COLOR_RGB2YCrCb)[:, :, 0]
        recovered_img_ycbcr = cv2.cvtColor(recovered_img, cv2.COLOR_RGB2YCrCb)[:, :, 0]
        clean_img_hsv = cv2.cvtColor(clean_img, cv2.COLOR_RGB2HSV)[:, :, 0]
        recovered_img_hsv = cv2.cvtColor(recovered_img, cv2.COLOR_RGB2HSV)[:, :, 0]
        clean_img_lab = cv2.cvtColor(clean_img, cv2.COLOR_RGB2Lab)[:, :, 0]
        recovered_img_lab = cv2.cvtColor(recovered_img, cv2.COLOR_RGB2Lab)[:, :, 0]
        # print(clean_img_lab)

        psnr_rgb += peak_signal_noise_ratio(clean_img[:, :, 2], recovered_img[:, :, 2], data_range=255)
        # ssim_rgb += structural_similarity(clean_img[:, :, 0], recovered_img[:, :, 0], data_range=255, multichannel=True, channel_axis=2)
        ssim_rgb += structural_similarity(clean_img[:, :, 2], recovered_img[:, :, 2], data_range=255, multichannel=False)

        psnr_ycbcr += peak_signal_noise_ratio(clean_img_ycbcr, recovered_img_ycbcr, data_range=255)
        ssim_ycbcr += structural_similarity(clean_img_ycbcr, recovered_img_ycbcr, data_range=255, multichannel=False)

        psnr_hsv += peak_signal_noise_ratio(clean_img_hsv, recovered_img_hsv, data_range=255)
        # ssim_hsv += structural_similarity(clean_img_hsv, recovered_img_hsv, data_range=255, multichannel=True, channel_axis=2)
        ssim_hsv += structural_similarity(clean_img_hsv, recovered_img_hsv, data_range=255, multichannel=False)

        psnr_lab += peak_signal_noise_ratio(clean_img_lab, recovered_img_lab, data_range=255)
        # ssim_lab += structural_similarity(clean_img_lab, recovered_img_lab, data_range=255, multichannel=True,
        #                                  channel_axis=2)
        ssim_lab += structural_similarity(clean_img_lab, recovered_img_lab, data_range=255, multichannel=False)

    return (psnr_rgb / recoverd.shape[0], ssim_rgb / recoverd.shape[0], psnr_ycbcr / recoverd.shape[0], ssim_ycbcr / recoverd.shape[0],
            psnr_hsv / recoverd.shape[0], ssim_hsv / recoverd.shape[0], psnr_lab / recoverd.shape[0], ssim_lab / recoverd.shape[0],
            recoverd.shape[0])


class timer():
    def __init__(self):
        self.acc = 0
        self.tic()

    def tic(self):
        self.t0 = time.time()

    def toc(self):
        return time.time() - self.t0

    def hold(self):
        self.acc += self.toc()

    def release(self):
        ret = self.acc
        self.acc = 0

        return ret

    def reset(self):
        self.acc = 0