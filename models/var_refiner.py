import math
from functools import partial
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

import dist
from models.basic_var import AdaLNBeforeHead, AdaLNSelfAttn, precompute_freqs_cis, precompute_freqs_cis_cross
from models.helpers import gumbel_softmax_with_rng, sample_with_top_k_top_p_
from models.vqvae import VQVAE, VectorQuantizer2


class SharedAdaLin(nn.Linear):
    def forward(self, cond_BD):
        C = self.weight.shape[0] // 6
        return super().forward(cond_BD).view(-1, 1, 6, C)  # B16C


class VAR(nn.Module):
    def __init__(
            self, vae_local: VQVAE,
            num_classes=1000, depth=16, embed_dim=1024, num_heads=16, mlp_ratio=4., drop_rate=0., attn_drop_rate=0.,
            drop_path_rate=0.,
            norm_eps=1e-6, shared_aln=False, cond_drop_rate=0.1,
            attn_l2_norm=False,
            patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),  # 10 steps by default
            flash_if_available=True, fused_if_available=True, rope_base=10000.0
    ):
        super().__init__()
        # 0. hyperparameters
        assert embed_dim % num_heads == 0
        self.Cvae, self.V = vae_local.Cvae, vae_local.vocab_size
        self.depth, self.C, self.D, self.num_heads = depth, embed_dim, embed_dim, num_heads

        self.cond_drop_rate = cond_drop_rate
        self.prog_si = -1  # progressive training

        self.patch_nums: Tuple[int] = patch_nums
        self.L = sum(pn ** 2 for pn in self.patch_nums)
        self.first_l = self.patch_nums[0] ** 2
        self.begin_ends = []
        cur = 0
        for i, pn in enumerate(self.patch_nums):
            self.begin_ends.append((cur, cur + pn ** 2))
            cur += pn ** 2

        self.num_stages_minus_1 = len(self.patch_nums) - 1
        self.rng = torch.Generator(device=dist.get_device())

        # 1. input (word) embedding
        quant: VectorQuantizer2 = vae_local.quantize
        self.vae_proxy: Tuple[VQVAE] = (vae_local,)
        self.vae_quant_proxy: Tuple[VectorQuantizer2] = (quant,)
        self.word_embed = nn.Linear(self.Cvae, self.C)

        # 2. class embedding
        init_std = math.sqrt(1 / self.C / 3)
        self.num_classes = num_classes
        self.uniform_prob = torch.full((1, num_classes), fill_value=1.0 / num_classes, dtype=torch.float32,
                                       device=dist.get_device())
        self.class_emb = nn.Embedding(self.num_classes + 1, self.C)
        nn.init.trunc_normal_(self.class_emb.weight.data, mean=0, std=init_std)

        self.pos_start = None

        self.pos_1LC = None

        # level embedding (similar to GPT's segment embedding, used to distinguish different levels of token pyramid)
        self.lvl_embed = nn.Embedding(len(self.patch_nums), self.C)
        nn.init.trunc_normal_(self.lvl_embed.weight.data, mean=0, std=init_std)

        # 4. backbone blocks
        self.shared_ada_lin = nn.Sequential(nn.SiLU(inplace=False),
                                            SharedAdaLin(self.D, 6 * self.C)) if shared_aln else nn.Identity()

        norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        self.drop_path_rate = drop_path_rate
        dpr = [x.item() for x in
               torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule (linearly increasing)
        self.blocks = nn.ModuleList([
            AdaLNSelfAttn(
                cond_dim=self.D, shared_aln=shared_aln,
                block_idx=block_idx, embed_dim=self.C, norm_layer=norm_layer, num_heads=num_heads, mlp_ratio=mlp_ratio,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[block_idx],
                last_drop_p=0 if block_idx == 0 else dpr[block_idx - 1],
                attn_l2_norm=attn_l2_norm,
                flash_if_available=flash_if_available, fused_if_available=fused_if_available,
            )
            for block_idx in range(depth)
        ])

        fused_add_norm_fns = [b.fused_add_norm_fn is not None for b in self.blocks]
        self.using_fused_add_norm_fn = any(fused_add_norm_fns)
        print(
            f'\n[constructor]  ==== flash_if_available={flash_if_available} ({sum(b.attn.using_flash for b in self.blocks)}/{self.depth}), fused_if_available={fused_if_available} (fusing_add_ln={sum(fused_add_norm_fns)}/{self.depth}, fusing_mlp={sum(b.ffn.fused_mlp_func is not None for b in self.blocks)}/{self.depth}) ==== \n'
            f'    [VAR config ] embed_dim={embed_dim}, num_heads={num_heads}, depth={depth}, mlp_ratio={mlp_ratio}\n'
            f'    [drop ratios ] drop_rate={drop_rate}, attn_drop_rate={attn_drop_rate}, drop_path_rate={drop_path_rate:g} ({torch.linspace(0, drop_path_rate, depth)})',
            end='\n\n', flush=True
        )

        # 5. attention mask used in training (for masking out the future)
        #    it won't be used in inference, since kv cache is enabled
        d: torch.Tensor = torch.cat([torch.full((pn * pn,), i) for i, pn in enumerate(self.patch_nums)]).view(1, self.L,
                                                                                                              1)
        dT = d.transpose(1, 2)  # dT: 11L
        lvl_1L = dT[:, 0].contiguous()
        self.register_buffer('lvl_1L', lvl_1L)
        attn_bias_for_masking = torch.where(d >= dT, 0., -torch.inf).reshape(1, 1, self.L, self.L)
        self.register_buffer('attn_bias_for_masking', attn_bias_for_masking.contiguous())

        # 6. classifier head
        self.head_nm = AdaLNBeforeHead(self.C, self.D, norm_layer=norm_layer)
        self.head = nn.Linear(self.C, self.V)

        # Layers for conditional tokens
        self.cond_layers = 1  # Number of VAE levels
        self.cond_layer_reso = patch_nums[-1]  # Final spatial dimension of VAE levels
        # Global embed for SOS
        self.global_embed = nn.Linear(32, self.C)
        self.global_embed_gain = nn.Parameter(torch.tensor(0.0), requires_grad=True)

        # Projectors for VAE encodings
        self.cond_proj4 = nn.Linear(self.Cvae, self.Cvae)

        self.word_embed_cond = nn.Linear(self.Cvae, self.C)  # Word embedding for cond

        # Replace with RoPE
        self.pos_1LC_cond = None

        # level embedding cond (similar to GPT's segment embedding, used to distinguish different levels of token pyramid)
        self.lvl_embed_cond = nn.Embedding(self.cond_layers, self.C)  # 4 levels of VAE
        nn.init.trunc_normal_(self.lvl_embed_cond.weight.data, mean=0, std=init_std)

        d: torch.Tensor = torch.cat(
            [torch.full((self.cond_layer_reso ** 2,), i) for i in range(self.cond_layers)]).view(1,
                                                                                                 self.cond_layers * self.cond_layer_reso ** 2,
                                                                                                 1)
        dT = d.transpose(1, 2)  # dT: 11L
        lvl_1L_cond = dT[:, 0].contiguous()
        self.register_buffer('lvl_1L_cond', lvl_1L_cond)

        # Add RoPE
        self.head_dim = self.C // self.num_heads
        freqs_cis = precompute_freqs_cis(dim=self.head_dim, patch_nums=patch_nums)
        freqs_cis_cross = precompute_freqs_cis_cross(dim=self.head_dim, patch_nums=[patch_nums[-1]],
                                                     theta=rope_base)

        self.register_buffer('freqs_cis', freqs_cis)
        self.register_buffer('freqs_cis_cross', freqs_cis_cross)

        # Refiner
        from models.vae_refiner_rope import Refiner

        self.vae_refiner = Refiner(seq_len=self.cond_layer_reso ** 2, dim=384, heads=6, mlp_dim=768)

    def get_logits(self, h_or_h_and_residual: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
                   cond_BD: Optional[torch.Tensor]):
        if not isinstance(h_or_h_and_residual, torch.Tensor):
            h, resi = h_or_h_and_residual  # fused_add_norm must be used
            h = resi + self.blocks[-1].drop_path(h)
        else:  # fused_add_norm is not used
            h = h_or_h_and_residual
        return self.head(self.head_nm(h.float(), cond_BD).float()).float()

    def autoregressive_infer_cfg(
            self, cond_BLC_wo_first_l,
    ) -> Tuple[torch.Tensor, float]:  # returns reconstructed image (B, 3, H, W) in [0, 1]

        ITen = torch.LongTensor
        B = cond_BLC_wo_first_l[3].shape[0]

        label_B = torch.full((B,), fill_value=self.num_classes).to(
            cond_BLC_wo_first_l[0].device)  # Fill all label_B to 1000
        sos = cond_BD = self.class_emb(label_B)

        fs_proj4 = self.cond_proj4(cond_BLC_wo_first_l[3].float())
        cond_BLC_proj = torch.cat([fs_proj4], dim=1)

        global_embed = self.global_embed_gain * self.global_embed(
            cond_BLC_proj.mean(dim=1, keepdim=True))  # Get global embedding
        cond_BD = cond_BD + global_embed.squeeze(1)  # Add it to cond_BD
        sos = sos.unsqueeze(1).expand(B, self.first_l, -1)
        sos = sos + global_embed

        lvl_pos = self.lvl_embed(self.lvl_1L)
        next_token_map = sos + lvl_pos[:, :self.first_l]
        cond_BLC = self.word_embed_cond(cond_BLC_proj.float())
        cond_BLC += self.lvl_embed_cond(self.lvl_1L_cond[:, :].expand(B, -1))

        cur_L = 0
        f_hat = sos.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1])

        all_blocks = []
        all_logits = []

        for b in self.blocks: b.attn.kv_caching(True)
        for si, pn in enumerate(self.patch_nums):  # si: i-th segment
            freqs_cis = self.freqs_cis[cur_L:cur_L + pn * pn]
            cur_L += pn * pn
            cond_BD_or_gss = self.shared_ada_lin(cond_BD)
            x = next_token_map

            AdaLNSelfAttn.forward
            for b in self.blocks:
                x = b(x=x, cond_BD=cond_BD_or_gss, attn_bias=None, condition=cond_BLC,
                      freqs_cis=freqs_cis,
                      freqs_cis_cross=self.freqs_cis_cross)
            all_blocks.append(x)
            logits_BlV = self.get_logits(x, cond_BD)
            all_logits.append(logits_BlV.clone())

            idx_Bl = logits_BlV.argmax(dim=-1)
            h_BChw = self.vae_quant_proxy[0].embedding(idx_Bl)  # B, l, Cvae

            h_BChw = h_BChw.transpose_(1, 2).reshape(B, self.Cvae, pn, pn)
            f_hat, next_token_map = self.vae_quant_proxy[0].get_next_autoregressive_input(si, len(self.patch_nums),
                                                                                          f_hat, h_BChw)
            if si != self.num_stages_minus_1:  # prepare for next stage
                next_token_map = next_token_map.view(B, self.Cvae, -1).transpose(1, 2)
                next_token_map = self.word_embed(next_token_map) + lvl_pos[:,
                cur_L:cur_L + self.patch_nums[si + 1] ** 2]

        for b in self.blocks: b.attn.kv_caching(False)

        B, C, H, W = f_hat.shape
        f_hat = f_hat.view(B, C, -1).transpose(1, 2)
        last_block = torch.cat(all_blocks, dim=1)
        f_hat = self.vae_refiner(f_hat, last_block)
        all_logits = torch.cat(all_logits, dim=1)

        return f_hat, all_logits

    def forward(self, f_hat, last_block) -> torch.Tensor:

        refined_fhat = self.vae_refiner(f_hat, last_block)

        return refined_fhat

    def debug_images(self, f_hat, save_dir, step):
        from torchvision.utils import save_image
        import os
        import datetime

        B = f_hat.shape[0]
        top_k = 900
        top_p = 0.95
        rng = None

        # Ensure the save directory exists
        os.makedirs(save_dir, exist_ok=True)

        # Reconstruct images using the VAE's decoder
        C, H, W = 32, 32, 32
        f_hat = f_hat.transpose(1, 2).view(B, C, H, W)
        reconstructed_imgs = self.vae_proxy[0].fhat_to_img(f_hat).add_(1).mul_(0.5)  # Shape: (B, 3, H_img, W_img)

        # Generate a unique timestamp for filenames
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Iterate over the batch and save each image individually
        for i in range(B):
            img = reconstructed_imgs[i].cpu()  # Move to CPU for saving
            if step is not None:
                img_filename = os.path.join(save_dir, f'image_step{step}_{timestamp}_{i}.png')
            else:
                img_filename = os.path.join(save_dir, f'image_{timestamp}_{i}.png')
            save_image(img, img_filename)
            print(f"Saved image: {img_filename}")  # Optional: Log saved image paths

        return reconstructed_imgs

    def init_weights(self, init_adaln=0.5, init_adaln_gamma=1e-5, init_head=0.02, init_std=0.02, conv_std_or_gain=0.02):
        if init_std < 0: init_std = (1 / self.C / 3) ** 0.5  # init_std < 0: automated

        print(f'[init_weights] {type(self).__name__} with {init_std=:g}')
        for m in self.modules():
            with_weight = hasattr(m, 'weight') and m.weight is not None
            with_bias = hasattr(m, 'bias') and m.bias is not None
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight.data, std=init_std)
                if with_bias: m.bias.data.zero_()
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight.data, std=init_std)
                if m.padding_idx is not None: m.weight.data[m.padding_idx].zero_()
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm,
                                nn.GroupNorm, nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
                if with_weight: m.weight.data.fill_(1.)
                if with_bias: m.bias.data.zero_()
            # conv: VAR has no conv, only VQVAE has conv
            elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d,
                                nn.ConvTranspose3d)):
                if conv_std_or_gain > 0:
                    nn.init.trunc_normal_(m.weight.data, std=conv_std_or_gain)
                else:
                    nn.init.xavier_normal_(m.weight.data, gain=-conv_std_or_gain)
                if with_bias: m.bias.data.zero_()

        if init_head >= 0:
            if isinstance(self.head, nn.Linear):
                self.head.weight.data.mul_(init_head)
                self.head.bias.data.zero_()
            elif isinstance(self.head, nn.Sequential):
                self.head[-1].weight.data.mul_(init_head)
                self.head[-1].bias.data.zero_()

        depth = len(self.blocks)
        for block_idx, sab in enumerate(self.blocks):
            sab: AdaLNSelfAttn
            sab.attn.proj.weight.data.div_(math.sqrt(2 * depth))
            sab.ffn.fc2.weight.data.div_(math.sqrt(2 * depth))
            if hasattr(sab.ffn, 'fcg') and sab.ffn.fcg is not None:
                nn.init.ones_(sab.ffn.fcg.bias)
                nn.init.trunc_normal_(sab.ffn.fcg.weight, std=1e-5)
            if hasattr(sab, 'ada_lin'):
                sab.ada_lin[-1].weight.data[2 * self.C:].mul_(init_adaln)
                sab.ada_lin[-1].weight.data[:2 * self.C].mul_(init_adaln_gamma)
                if hasattr(sab.ada_lin[-1], 'bias') and sab.ada_lin[-1].bias is not None:
                    sab.ada_lin[-1].bias.data.zero_()
            elif hasattr(sab, 'ada_gss'):
                sab.ada_gss.data[:, :, 2:].mul_(init_adaln)
                sab.ada_gss.data[:, :, :2].mul_(init_adaln_gamma)

    def extra_repr(self):
        return f'drop_path_rate={self.drop_path_rate:g}'
