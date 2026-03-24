import time
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

import dist
from models import VQVAE, VectorQuantizer2
from models.var_refiner import VAR
from utils.amp_sc import AmpOptimizer
from utils.misc import MetricLogger, TensorboardLogger
import wandb
from skimage.metrics import structural_similarity as ssim_skimage
import torch.nn.functional as F

Ten = torch.Tensor
FTen = torch.Tensor
ITen = torch.LongTensor
BTen = torch.BoolTensor


class VARTrainer(object):
    def __init__(
            self, device, patch_nums: Tuple[int, ...], resos: Tuple[int, ...],
            vae_local: VQVAE, var_wo_ddp: VAR, var: DDP,
            var_opt: AmpOptimizer, label_smooth: float,
    ):
        super(VARTrainer, self).__init__()

        self.var, self.vae_local, self.quantize_local = var, vae_local, vae_local.quantize
        self.quantize_local: VectorQuantizer2
        self.var_wo_ddp: VAR = var_wo_ddp  # after torch.compile
        self.var_opt = var_opt

        del self.var_wo_ddp.rng
        self.var_wo_ddp.rng = torch.Generator(device=device)

        self.label_smooth = label_smooth
        self.train_loss = nn.CrossEntropyLoss(label_smoothing=label_smooth, reduction='none')
        self.val_loss = nn.CrossEntropyLoss(label_smoothing=0.0, reduction='mean')
        self.L = sum(pn * pn for pn in patch_nums)
        self.last_l = patch_nums[-1] * patch_nums[-1]
        self.loss_weight = torch.ones(1, self.L, device=device) / self.L

        self.patch_nums, self.resos = patch_nums, resos
        self.begin_ends = []
        cur = 0
        for i, pn in enumerate(patch_nums):
            self.begin_ends.append((cur, cur + pn * pn))
            cur += pn * pn

        self.prog_it = 0
        self.last_prog_si = -1
        self.first_prog = True

    def compute_ssim_skimage(self, tensor1: torch.Tensor, tensor2: torch.Tensor) -> float:
        # Move tensors to CPU and convert to numpy arrays
        arr1 = tensor1.detach().cpu().numpy()
        arr2 = tensor2.detach().cpu().numpy()

        B, C, H, W = arr1.shape
        ssim_total = 0.0

        for b in range(B):
            assert C > 1, "Channel size is not 3"
            if C == 1:
                # For single-channel (grayscale) images
                img1 = arr1[b, 0]
                img2 = arr2[b, 0]
                # data_range is the difference between max and min values of the first image.
                ssim_val = ssim_skimage(img1, img2, data_range=img1.max() - img1.min())
            else:
                # For multi-channel images, transpose to (H, W, C)
                img1 = arr1[b].transpose(1, 2, 0)
                img2 = arr2[b].transpose(1, 2, 0)
                ssim_val = ssim_skimage(img1, img2, data_range=1.0, channel_axis=2)
            ssim_total += ssim_val

        return ssim_total / B

    def psnr_tensor(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        Get PSNR between two tensors. Used for Validations
        """
        mse = torch.mean((X - Y) ** 2)
        # Avoid division by zero if mse is zero:
        if mse == 0:
            return torch.tensor(float('inf'))
        psnr = 10 * torch.log10(1 / mse)
        return psnr

    @torch.no_grad()
    def eval_ep(self, ld_val: DataLoader):
        tot = 0
        L_mean, L_tail, acc_mean, acc_tail = 0, 0, 0, 0
        psnr_mean, ssim_mean = 0.0, 0.0
        val_loss = 0.0
        stt = time.time()
        training = self.var_wo_ddp.training
        self.var_wo_ddp.eval()
        for inp_B3HW, degrad_B3HW, label_B, dataset_restoration, sample_idx in ld_val:  # Added degraded image and dataset name to unpack
            B, V = label_B.shape[0], self.vae_local.vocab_size
            inp_B3HW = inp_B3HW.to(dist.get_device(), non_blocking=True)
            label_B = label_B.to(dist.get_device(), non_blocking=True)
            degrad_B3HW = degrad_B3HW.to(dist.get_device(), non_blocking=True)  # Move degraded

            gt_idx_Bl: List[ITen] = self.vae_local.img_to_idxBl(inp_B3HW)
            gt_BL = torch.cat(gt_idx_Bl, dim=1)  # To calculate loss
            x_BLCv_wo_first_l: Ten = self.quantize_local.idxBl_to_var_input(gt_idx_Bl)

            gt_continuous = self.vae_local.img_to_f(inp_B3HW)  # B x 32 x 16 x 16, to predict from discrete tokens
            B, C, H, W = gt_continuous.shape
            gt_continuous = gt_continuous.view(B, C, -1).transpose(1, 2)

            cond_img_inp = self.vae_local.img_to_fs(degrad_B3HW)

            refined, logits_BLV = self.var_wo_ddp.autoregressive_infer_cfg(
                cond_BLC_wo_first_l=cond_img_inp)  # degraded image goes as input, also var returns f_hat
            val_loss += F.l1_loss(refined, gt_continuous, reduction='mean')
            # Decode the images for PSNR calculation
            restored = self.var_wo_ddp.debug_images(refined, save_dir='val_images/', step=tot)
            inp_B3HW.add_(1).mul_(0.5)
            # acc_mean += self.psnr_tensor(restored, inp_B3HW) * B  # Multiply by B as mean is calculated already
            psnr_mean += self.psnr_tensor(restored, inp_B3HW) * B  # Multiply by B as mean is calculated already
            ssim_mean += self.compute_ssim_skimage(restored, inp_B3HW) * B

            L_mean += self.val_loss(logits_BLV.data.view(-1, V), gt_BL.view(-1)) * B
            L_tail += self.val_loss(logits_BLV.data[:, -self.last_l:].reshape(-1, V),
                                    gt_BL[:, -self.last_l:].reshape(-1)) * B
            acc_mean += (logits_BLV.data.argmax(dim=-1) == gt_BL).sum() * (
                        100 / gt_BL.shape[1])  # Treat acc_mean as PSNR
            acc_tail += (logits_BLV.data[:, -self.last_l:].argmax(dim=-1) == gt_BL[:, -self.last_l:]).sum() * (
                        100 / self.last_l)
            tot += B
        self.var_wo_ddp.train(training)
        if L_mean == 0:
            L_mean, L_tail, acc_mean, acc_tail = torch.tensor(L_mean), torch.tensor(L_tail), torch.tensor(
                acc_mean), torch.tensor(acc_tail)

        stats = L_mean.new_tensor([L_mean.item(), L_tail.item(), acc_mean.item(), acc_tail.item(), psnr_mean.item(),
                                   ssim_mean.item(), val_loss.item(), tot])
        dist.allreduce(stats)
        tot = round(stats[-1].item())
        stats /= tot
        L_mean, L_tail, acc_mean, acc_tail, psnr_mean, ssim_mean, val_loss, _ = stats.tolist()
        # Log to wandb
        if dist.is_master():
            wandb.log({
                "Val Loss Mean (Lm)": L_mean,
                "Val Loss Tail (Lm)": L_tail,
                "Val Acc Mean (Accm)": acc_mean,
                "Val Acc Tail (Acct)": acc_tail,
                "Val PSNR": psnr_mean,
                "Val SSIM": ssim_mean,
                "Val refiner loss": val_loss
            })
        return L_mean, L_tail, acc_mean, acc_tail, tot, time.time() - stt

    def train_step(
            self, it: int, g_it: int, stepping: bool, metric_lg: MetricLogger, tb_lg: TensorboardLogger,
            inp_B3HW: FTen, degrad_B3HW: FTen, label_B: Union[ITen, FTen], prog_si: int,
            prog_wp_it: float,
    ) -> Tuple[Optional[Union[Ten, float]], Optional[float]]:  # take degraded image as input
        # if progressive training
        self.var_wo_ddp.prog_si = self.vae_local.quantize.prog_si = prog_si
        if self.last_prog_si != prog_si:
            if self.last_prog_si != -1: self.first_prog = False
            self.last_prog_si = prog_si
            self.prog_it = 0
        self.prog_it += 1
        prog_wp = max(min(self.prog_it / prog_wp_it, 1), 0.01)
        if self.first_prog: prog_wp = 1  # no prog warmup at first prog stage, as it's already solved in wp
        if prog_si == len(self.patch_nums) - 1: prog_si = -1  # max prog, as if no prog

        # forward
        B, V = label_B.shape[0], self.vae_local.vocab_size
        self.var.require_backward_grad_sync = stepping

        gt_continuous = self.vae_local.img_to_f(inp_B3HW)  # B x 32 x 16 x 16, to predict from discrete tokens
        B, C, H, W = gt_continuous.shape
        gt_continuous = gt_continuous.view(B, C, -1).transpose(1, 2)
        # Get degraded image as tokens
        cond_img_inp = self.vae_local.img_to_fs(degrad_B3HW)

        with self.var_opt.amp_ctx:
            # self.var_wo_ddp.forward
            refined, logits_BLV = self.var_wo_ddp.autoregressive_infer_cfg(
                cond_img_inp)  # Input is now degrade image, not clean image, also VAR returns f_hat

            loss = F.l1_loss(refined, gt_continuous, reduction='mean')

        # backward
        grad_norm, scale_log2 = self.var_opt.backward_clip_step(loss=loss, stepping=stepping)

        # log
        if it == 0 or it in metric_lg.log_iters:
            grad_norm = grad_norm.item()

            # Log to wandb
            if dist.is_master():
                wandb.log({
                    "Refinerloss": loss,
                    # "Cos los": cos_loss,
                    "Grad norm": grad_norm
                })

        self.var_wo_ddp.prog_si = self.vae_local.quantize.prog_si = -1
        return grad_norm, scale_log2

    def get_config(self):
        return {
            'patch_nums': self.patch_nums, 'resos': self.resos,
            'label_smooth': self.label_smooth,
            'prog_it': self.prog_it, 'last_prog_si': self.last_prog_si, 'first_prog': self.first_prog,
        }

    def state_dict(self):
        state = {'config': self.get_config()}
        for k in ('var_wo_ddp', 'vae_local', 'var_opt'):
            m = getattr(self, k)
            if m is not None:
                if hasattr(m, '_orig_mod'):
                    m = m._orig_mod
                state[k] = m.state_dict()
        return state

    def load_state_dict(self, state, strict=True, skip_vae=False):
        for k in ('var_wo_ddp', 'vae_local', 'var_opt'):
            if skip_vae and 'vae' in k: continue
            m = getattr(self, k)
            if m is not None:
                if hasattr(m, '_orig_mod'):
                    m = m._orig_mod
                ret = m.load_state_dict(state[k], strict=strict)
                if ret is not None:
                    missing, unexpected = ret
                    print(f'[VARTrainer.load_state_dict] {k} missing:  {missing}')
                    print(f'[VARTrainer.load_state_dict] {k} unexpected:  {unexpected}')

        config: dict = state.pop('config', None)
        self.prog_it = config.get('prog_it', 0)
        self.last_prog_si = config.get('last_prog_si', -1)
        self.first_prog = config.get('first_prog', True)
        if config is not None:
            for k, v in self.get_config().items():
                if config.get(k, None) != v:
                    err = f'[VAR.load_state_dict] config mismatch:  this.{k}={v} (ckpt.{k}={config.get(k, None)})'
                    if strict:
                        raise AttributeError(err)
                    else:
                        print(err)

    def normalize_to_imagenet(self, tensor):
        """
        New function. Normalize a batch of images from range [-1, 1] to ImageNet mean/std normalized space.

        Args:
            tensor (torch.Tensor): Input tensor of shape [B, C, H, W] with pixel values in [-1, 1].

        Returns:
            torch.Tensor: Normalized tensor of shape [B, C, H, W].
        """
        # Rescale from [-1, 1] to [0, 1]
        tensor = (tensor + 1.0) / 2.0

        # ImageNet mean and std for 3 channels (C=3)
        mean = torch.tensor([0.485, 0.456, 0.406], device=tensor.device, dtype=tensor.dtype)
        std = torch.tensor([0.229, 0.224, 0.225], device=tensor.device, dtype=tensor.dtype)

        # Reshape mean and std for broadcasting: shape becomes [1, C, 1, 1]
        mean = mean.view(1, -1, 1, 1)
        std = std.view(1, -1, 1, 1)

        # Normalize using ImageNet statistics
        tensor = (tensor - mean) / std
        return tensor

    def save_side_by_side(self,
                          inp_B3HW: torch.Tensor,
                          decoded_cont_gt: torch.Tensor,
                          decoded_disc_gt: torch.Tensor,
                          decoded_disc_direct: torch.Tensor,
                          out_prefix: str = "output"
                          ):
        """
        Saves four images side-by-side:
            1) inp_B3HW (original input, -1..1) -> scaled to 0..1
            2) decoded_cont_gt (already 0..1)
            3) decoded_disc_gt (already 0..1)
            4) decoded_disc_direct (already 0..1)
        For each sample in the batch, saves a PNG image.

        Args:
            inp_B3HW:            [B, 3, H, W], values in [-1,1]
            decoded_cont_gt:     [B, 3, H, W], values in [0,1]
            decoded_disc_gt:     [B, 3, H, W], values in [0,1]
            decoded_disc_direct: [B, 3, H, W], values in [0,1]
            out_prefix:          filename prefix for saved images
        """
        import torchvision.utils as vutils
        B = inp_B3HW.shape[0]

        for b_idx in range(B):
            # 1) Take each image in the batch
            # Scale inp_B3HW from [-1,1] -> [0,1]
            inp_img = 0.5 * (inp_B3HW[b_idx] + 1.0)
            inp_img = inp_img.clamp(0, 1)

            cont_img = decoded_cont_gt[b_idx].clamp(0, 1)
            disc_img = decoded_disc_gt[b_idx].clamp(0, 1)
            direct_img = decoded_disc_direct[b_idx].clamp(0, 1)

            # 2) Stack them into a single tensor of shape [4, 3, H, W]
            #    in the order you specified
            all_imgs = torch.stack([inp_img, cont_img, disc_img, direct_img], dim=0)

            # 3) Make a grid with 4 images in one row: nrow=4
            grid = vutils.make_grid(all_imgs, nrow=4)

            # 4) Save to file
            out_path = f"{out_prefix}_sample{b_idx}.png"
            vutils.save_image(grid, out_path)
            print(f"Saved {out_path}")