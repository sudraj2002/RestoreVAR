import os
import torch, torchvision
from torch.utils.data import DataLoader
import random
import numpy as np
import PIL.Image as PImage, PIL.ImageDraw as PImageDraw
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)     # disable default parameter init for faster speed
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)  # disable default parameter init for faster speed
from models import VQVAE, build_vae_var
from utils.data import build_dataset
import time
from all_args import get_args

from metric_utils.val_utils import compute_psnr_ssim, AverageMeter
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


def get_vae_output_direct(var, cond_img):
    f = var.vae_proxy[0].quant_conv(var.vae_proxy[0].encoder(cond_img))
    cond_img_gt_idx_Bl = var.vae_proxy[0].quantize.f_to_idxBl_or_fhat(f, to_fhat=True, v_patch_nums=patch_nums)
    vae_ops = var.vae_proxy[0].fhat_to_img(cond_img_gt_idx_Bl[-1]).add_(1).mul_(0.5)
    return vae_ops, cond_img_gt_idx_Bl[-1]

def get_vae_output_cont(var, cond_img):
    f = var.vae_proxy[0].quant_conv(var.vae_proxy[0].encoder(cond_img))
    vae_ops = var.vae_proxy[0].fhat_to_img(f).add_(1).mul_(0.5)
    return vae_ops, f

def count_params(module, trainable_only=True):
    if trainable_only:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in module.parameters())

if __name__ == '__main__':
    args = get_args()

    MODEL_DEPTH = args.model_depth
    json_path = args.json_path
    patch_nums = tuple(args.patch_nums)
    vae_ckpt = args.vae_ckpt
    var_ckpt = args.var_ckpt

    result_dir = args.result_dir
    calc_metrics = args.calc_metrics
    seed = args.seed
    MODEL_DEPTH = 16  # Change if you train larger RestoreVAR
    assert MODEL_DEPTH in {16, 20, 24, 30}

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if 'vae' not in globals() or 'var' not in globals():
        vae, var = build_vae_var(
            V=4096, Cvae=32, ch=160, share_quant_resi=4,    # hard-coded VQVAE hyperparameters
            device=device, patch_nums=patch_nums,
            num_classes=1000, depth=MODEL_DEPTH, shared_aln=False, inference_mode=True
        )

    # load checkpoints
    vae.load_state_dict(torch.load(vae_ckpt, map_location='cpu'), strict=True)
    var_loaded = torch.load(var_ckpt, map_location='cpu')
    var.load_state_dict(var_loaded, strict=True)
    vae.eval(), var.eval()

    for p in vae.parameters(): p.requires_grad_(False)
    for p in var.parameters(): p.requires_grad_(False)
    print(f'prepare finished.')

    if result_dir == '':
        result_dir = 'results/'

    print(f"Starting testing: ckpt: {var_ckpt}, Depth: {MODEL_DEPTH}, "
          f"Resolution: {patch_nums[-1] ** 2}")

    if calc_metrics:
        lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type='alex').to(device)

    data_load_reso = max(patch_nums) * 16 # Patch size=16
    mid_reso = 1.125
    total_classes, _, test_set = build_dataset(json_path, json_path, final_reso=data_load_reso, mid_reso=mid_reso)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    # set args
    stop_count = 5000 # Large datasets take a long time

    # seed
    if seed:
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # run faster
    tf32 = True
    torch.backends.cudnn.allow_tf32 = bool(tf32)
    torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
    torch.set_float32_matmul_precision('high' if tf32 else 'highest')

    # Metrics
    if calc_metrics:
        psnr = AverageMeter()
        ssim = AverageMeter()
        lpips_scores = AverageMeter()
        lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type='alex').cuda()

    total_time = 0
    for i, data in enumerate(test_loader):
        if i > stop_count:
            break

        gt_img, cond_img, class_label, ds, name = data
        ds = ds[0]
        name = name[0]

        if ds == 'REVIDE':
            name = name.split('/')[-2] + '_' + name.split('/')[-1]
        elif ds == 'GoPro':
            name = name.split('/')[-3] + '_' + name.split('/')[-1]
        else:
            name = name.split('/')[-1]

        cond_img = cond_img.to(device)
        gt_img = gt_img.to(device)

        # sample
        B = 1
        label_B: torch.LongTensor = torch.tensor(class_label, device=device)
        with torch.inference_mode():
            # TODO: dtype torch.float16 throws error
            with torch.autocast('cuda', enabled=True, dtype=torch.float32, cache_enabled=True):
                st = time.time() # using bfloat16 can be faster

                recon_B3HW, cont_fhat = var.autoregressive_infer_cfg(B=B, cond_img=cond_img,
                                                               g_seed=seed)

                et = time.time()
                total_time += (et - st)

                if calc_metrics:
                    pred_tensor = recon_B3HW.clone()
                    gt_tensor = (gt_img + 1.0) / 2.0

                    temp_psnr, temp_ssim, N = compute_psnr_ssim(
                        pred_tensor.to(torch.float32),
                        gt_tensor.to(torch.float32)
                    )
                    psnr.update(temp_psnr, N)
                    ssim.update(temp_ssim, N)

                    # Prepare LPIPS inputs
                    restored_lpips = 2 * pred_tensor - 1
                    clean_patch_lpips = 2 * gt_tensor - 1
                    lpips_value = lpips_metric(restored_lpips, clean_patch_lpips)
                    lpips_scores.update(lpips_value.item(), N)

        # Comment if you just want to save model output
        recon_B3HW = torch.cat([((cond_img + 1) * 0.5).to(device), recon_B3HW,
                                ((gt_img + 1) * 0.5).to(device)], dim=0)
        chw = torchvision.utils.make_grid(recon_B3HW.clone(), nrow=8, padding=0, pad_value=1.0)
        chw = chw.permute(1, 2, 0).mul_(255).cpu().numpy()
        chw = PImage.fromarray(chw.astype(np.uint8))
        os.makedirs(result_dir + ds, exist_ok=True)
        if 'CDD' in ds or 'LOLBlur' in ds:
            # For reproducing paper metrics
            chw.save(os.path.join(result_dir, ds, name.replace('.png', '.jpg').replace('.tif', '.jpg')), format="JPEG",
                    quality=95)
        else:
            chw.save(os.path.join(result_dir, ds, name))

    print(f"Finished evaluation; Average time taken: {total_time / (i + 1)}, Depth: {MODEL_DEPTH}, "
          f"Resolution: {patch_nums[-1] ** 2}, ckpt: {var_ckpt}")
    if calc_metrics:
        print("psnr: %.2f, ssim: %.3f, lpips: %.3f" %
              (psnr.avg, ssim.avg, lpips_scores.avg))