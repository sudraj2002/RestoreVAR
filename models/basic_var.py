"""
VAR main functions
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.helpers import DropPath, drop_path
from einops import rearrange

# this file only provides the 3 blocks used in VAR transformer
__all__ = ['FFN', 'AdaLNSelfAttn', 'AdaLNBeforeHead']

# automatically import fused operators
dropout_add_layer_norm = fused_mlp_func = memory_efficient_attention = flash_attn_func = None
try:
    from flash_attn.ops.layer_norm import dropout_add_layer_norm
    from flash_attn.ops.fused_dense import fused_mlp_func
except ImportError:
    pass
# automatically import faster attention implementations
try:
    from xformers.ops import memory_efficient_attention
except ImportError:
    pass
try:
    from flash_attn import flash_attn_func  # qkv: BLHc, ret: BLHcq
except ImportError:
    pass
try:
    from torch.nn.functional import scaled_dot_product_attention as slow_attn  # q, k, v: BHLc
except ImportError:
    def slow_attn(query, key, value, scale: float, attn_mask=None, dropout_p=0.0):
        attn = query.mul(scale) @ key.transpose(-2, -1)  # BHLc @ BHcL => BHLL
        if attn_mask is not None: attn.add_(attn_mask)
        return (F.dropout(attn.softmax(dim=-1), p=dropout_p, inplace=True) if dropout_p > 0 else attn.softmax(
            dim=-1)) @ value


# RoPE code, taken from VarSR (https://github.com/quyp2000/VARSR)
def precompute_freqs_cis_cross(dim: int, patch_nums, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    t_x = []
    t_y = []
    grid = patch_nums[-1]
    index = torch.meshgrid(torch.arange(grid), torch.arange(grid))
    t_x.append(index[0].flatten() / grid * grid)
    t_y.append(index[1].flatten() / grid * grid)

    t_x = torch.cat(t_x, 0)
    t_y = torch.cat(t_y, 0)
    freqs_x = torch.outer(t_x, freqs).float()  # type: ignore
    freqs_y = torch.outer(t_y, freqs).float()  # type: ignore
    freqs_cis_x = torch.polar(torch.ones_like(freqs_x), freqs_x)  # complex64
    freqs_cis_y = torch.polar(torch.ones_like(freqs_y), freqs_y)
    freqs_cis = torch.cat([freqs_cis_x, freqs_cis_y], 1)
    return freqs_cis


def precompute_freqs_cis(dim: int, patch_nums, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    t_x = []
    t_y = []
    grid = patch_nums[-1]
    for patch_num in patch_nums:
        index = torch.meshgrid(torch.arange(patch_num), torch.arange(patch_num))
        t_x.append(index[0].flatten() / patch_num * grid)
        t_y.append(index[1].flatten() / patch_num * grid)

    t_x = torch.cat(t_x, 0)
    t_y = torch.cat(t_y, 0)
    freqs_x = torch.outer(t_x, freqs).float()  # type: ignore
    freqs_y = torch.outer(t_y, freqs).float()  # type: ignore
    freqs_cis_x = torch.polar(torch.ones_like(freqs_x), freqs_x)  # complex64
    freqs_cis_y = torch.polar(torch.ones_like(freqs_y), freqs_y)
    freqs_cis = torch.cat([freqs_cis_x, freqs_cis_y], 1)
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    if freqs_cis.ndim < 3:
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    else:
        shape = [d if i == 1 or i == ndim - 1 or i == 0 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
        xq: torch.Tensor,
        xk: torch.Tensor,
        freqs_cis: torch.Tensor,
):
    xq, xk = xq.permute(0, 2, 1, 3), xk.permute(0, 2, 1, 3)
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq).permute(0, 2, 1, 3), xk_out.type_as(xk).permute(0, 2, 1, 3)


def apply_rotary_emb_cross(
        xq: torch.Tensor,
        xk: torch.Tensor,
        freqs_cis: torch.Tensor,
        freqs_cis_cross: torch.Tensor,
):
    xq, xk = xq.permute(0, 2, 1, 3), xk.permute(0, 2, 1, 3)
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis_cross = reshape_for_broadcast(freqs_cis_cross, xk_)
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis_cross).flatten(3)
    return xq_out.type_as(xq).permute(0, 2, 1, 3), xk_out.type_as(xk).permute(0, 2, 1, 3)


class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0., fused_if_available=True):
        super().__init__()
        self.fused_mlp_func = fused_mlp_func if fused_if_available else None
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU(approximate='tanh')
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop, inplace=True) if drop > 0 else nn.Identity()

    def forward(self, x):
        if self.fused_mlp_func is not None:
            return self.drop(self.fused_mlp_func(
                x=x, weight1=self.fc1.weight, weight2=self.fc2.weight, bias1=self.fc1.bias, bias2=self.fc2.bias,
                activation='gelu_approx', save_pre_act=self.training, return_residual=False, checkpoint_lvl=0,
                heuristic=0, process_group=None,
            ))
        else:
            return self.drop(self.fc2(self.act(self.fc1(x))))

    def extra_repr(self) -> str:
        return f'fused_mlp_func={self.fused_mlp_func is not None}'


class CrossAttention(nn.Module):
    def __init__(
            self, block_idx, embed_dim=768, num_heads=12,
            attn_drop=0., proj_drop=0., attn_l2_norm=False, flash_if_available=True
    ):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.block_idx, self.num_heads, self.head_dim = block_idx, num_heads, embed_dim // num_heads
        self.attn_l2_norm = attn_l2_norm
        self.scale = 0.25 / math.sqrt(self.head_dim) if not attn_l2_norm else 1.0
        if attn_l2_norm:
            self.scale_mul_1H11 = nn.Parameter(torch.full((1, num_heads, 1, 1), fill_value=4.0).log(),
                                               requires_grad=True)
            self.max_scale_mul = torch.log(torch.tensor(100.)).item()

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.kv_proj = nn.Linear(embed_dim, embed_dim * 2, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.out_dropout = nn.Dropout(proj_drop) if proj_drop > 0 else nn.Identity()
        self.attn_drop = attn_drop
        self.using_flash = flash_if_available and flash_attn_func is not None
        self.using_xform = flash_if_available and memory_efficient_attention is not None
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        self.gain = nn.Parameter(torch.tensor(0.0), requires_grad=True)

        self.using_flash = False
        self.using_xform = False

    def forward(self, x_q, x_kv, freqs_cis, freqs_cis_cross, attn_bias=None):
        # Normalize
        q = self.norm_q(x_q)
        kv = self.norm_kv(x_kv)

        # Projections
        q = self.q_proj(q)
        k, v = self.kv_proj(kv).chunk(2, dim=-1)

        q = rearrange(q, 'b l (h d) -> b h l d', h=self.num_heads)
        k = rearrange(k, 'b l (h d) -> b h l d', h=self.num_heads)
        v = rearrange(v, 'b l (h d) -> b h l d', h=self.num_heads)

        # L2 norm attention if enabled
        if self.attn_l2_norm:
            scale_mul = self.scale_mul_1H11.clamp_max(self.max_scale_mul).exp()
            q = F.normalize(q, dim=-1) * scale_mul
            k = F.normalize(k, dim=-1)

        # Apply RoPE
        q, k = apply_rotary_emb_cross(q, k, freqs_cis=freqs_cis, freqs_cis_cross=freqs_cis_cross)

        # Choose backend
        dropout_p = self.attn_drop if self.training else 0.0
        dtype = q.dtype

        if self.using_flash:
            assert False
            out = flash_attn_func(q.to(dtype), k.to(dtype), v.to(dtype), dropout_p=dropout_p, softmax_scale=self.scale)
        elif self.using_xform:
            assert False
            out = memory_efficient_attention(q.to(dtype), k.to(dtype), v.to(dtype), attn_bias=None, p=dropout_p,
                                             scale=self.scale)
        else:
            attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
            if attn_bias is not None:
                attn = attn + attn_bias
            attn = F.softmax(attn, dim=-1)
            if dropout_p > 0:
                attn = F.dropout(attn, p=dropout_p, training=self.training)
            out = torch.matmul(attn, v)

        out = rearrange(out, 'b h l d -> b l (h d)')
        return self.out_dropout(self.gain * self.out_proj(out))

    def extra_repr(self) -> str:
        return f'block={self.block_idx}, dim={self.head_dim * self.num_heads}, heads={self.num_heads}, attn_l2_norm={self.attn_l2_norm}, using_flash={self.using_flash}'


class SelfAttention(nn.Module):
    def __init__(
            self, block_idx, embed_dim=768, num_heads=12,
            attn_drop=0., proj_drop=0., attn_l2_norm=False, flash_if_available=True
    ):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.block_idx, self.num_heads, self.head_dim = block_idx, num_heads, embed_dim // num_heads  # =64
        self.attn_l2_norm = attn_l2_norm
        if self.attn_l2_norm:
            self.scale = 1
            self.scale_mul_1H11 = nn.Parameter(torch.full(size=(1, self.num_heads, 1, 1), fill_value=4.0).log(),
                                               requires_grad=True)
            self.max_scale_mul = torch.log(torch.tensor(100)).item()
        else:
            self.scale = 0.25 / math.sqrt(self.head_dim)

        self.mat_qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.q_bias, self.v_bias = nn.Parameter(torch.zeros(embed_dim)), nn.Parameter(torch.zeros(embed_dim))
        self.register_buffer('zero_k_bias', torch.zeros(embed_dim))

        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop, inplace=True) if proj_drop > 0 else nn.Identity()
        self.attn_drop: float = attn_drop
        self.using_flash = flash_if_available and flash_attn_func is not None
        self.using_xform = flash_if_available and memory_efficient_attention is not None

        self.using_flash = False
        self.using_xform = False

        # only used during inference
        self.caching, self.cached_k, self.cached_v = False, None, None

    def kv_caching(self, enable: bool):
        self.caching, self.cached_k, self.cached_v = enable, None, None

    # NOTE: attn_bias is None during inference because kv cache is enabled
    def forward(self, x, attn_bias, freqs_cis):
        B, L, C = x.shape

        qkv = F.linear(input=x, weight=self.mat_qkv.weight,
                       bias=torch.cat((self.q_bias, self.zero_k_bias, self.v_bias))).view(B, L, 3, self.num_heads,
                                                                                          self.head_dim)
        main_type = qkv.dtype
        # qkv: BL3Hc

        using_flash = self.using_flash and attn_bias is None and qkv.dtype != torch.float32
        if using_flash or self.using_xform:
            q, k, v = qkv.unbind(dim=2);
            dim_cat = 1  # q or k or v: BLHc
        else:
            q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0);
            dim_cat = 2  # q or k or v: BHLc

        if self.attn_l2_norm:
            scale_mul = self.scale_mul_1H11.clamp_max(self.max_scale_mul).exp()
            if using_flash or self.using_xform: scale_mul = scale_mul.transpose(1, 2)  # 1H11 to 11H1
            q = F.normalize(q, dim=-1).mul(scale_mul)
            k = F.normalize(k, dim=-1)

        # Apply RoPE
        q, k = apply_rotary_emb(q, k, freqs_cis=freqs_cis)

        if self.caching:
            if self.cached_k is None:
                self.cached_k = k;
                self.cached_v = v
            else:
                k = self.cached_k = torch.cat((self.cached_k, k), dim=dim_cat);
                v = self.cached_v = torch.cat(
                    (self.cached_v, v), dim=dim_cat)

        dropout_p = self.attn_drop if self.training else 0.0
        if using_flash:
            oup = flash_attn_func(q.to(dtype=main_type), k.to(dtype=main_type), v.to(dtype=main_type),
                                  dropout_p=dropout_p, softmax_scale=self.scale).view(B, L, C)
        elif self.using_xform:
            oup = memory_efficient_attention(q.to(dtype=main_type), k.to(dtype=main_type), v.to(dtype=main_type),
                                             attn_bias=None if attn_bias is None else attn_bias.to(
                                                 dtype=main_type).expand(B, self.num_heads, -1, -1), p=dropout_p,
                                             scale=self.scale).view(B, L, C)
        else:
            oup = slow_attn(query=q, key=k, value=v, scale=self.scale, attn_mask=attn_bias,
                            dropout_p=dropout_p).transpose(1, 2).reshape(B, L, C)

        return self.proj_drop(self.proj(oup))

    def extra_repr(self) -> str:
        return f'using_flash={self.using_flash}, using_xform={self.using_xform}, attn_l2_norm={self.attn_l2_norm}'


class AdaLNSelfAttn(nn.Module):
    """
    No AdaLN is used.
    """
    def __init__(
            self, block_idx, last_drop_p, embed_dim, cond_dim, shared_aln: bool, norm_layer,
            num_heads, mlp_ratio=4., drop=0., attn_drop=0., drop_path=0., attn_l2_norm=False,
            flash_if_available=False, fused_if_available=True
    ):
        super(AdaLNSelfAttn, self).__init__()
        self.block_idx, self.last_drop_p, self.C = block_idx, last_drop_p, embed_dim
        self.C, self.D = embed_dim, cond_dim
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.attn = SelfAttention(block_idx=block_idx, embed_dim=embed_dim, num_heads=num_heads, attn_drop=attn_drop,
                                  proj_drop=drop,
                                  attn_l2_norm=attn_l2_norm, flash_if_available=flash_if_available)
        self.ffn = FFN(in_features=embed_dim, hidden_features=round(embed_dim * mlp_ratio), drop=drop,
                       fused_if_available=fused_if_available)

        self.ln_wo_grad = norm_layer(embed_dim, elementwise_affine=False)
        self.shared_aln = shared_aln

        self.fused_add_norm_fn = None

        self.cross_attn = CrossAttention(block_idx=block_idx, embed_dim=embed_dim, num_heads=num_heads,
                                         attn_drop=attn_drop, proj_drop=drop,
                                         attn_l2_norm=attn_l2_norm, flash_if_available=flash_if_available)

    # NOTE: attn_bias is None during inference because kv cache is enabled
    def forward(self, x, cond_BD, attn_bias, condition, freqs_cis, freqs_cis_cross):  # C: embed_dim, D: cond_dim
        x = x + self.drop_path(
            self.attn(self.ln_wo_grad(x), attn_bias=attn_bias, freqs_cis=freqs_cis))

        x = x + self.drop_path(
            self.ffn(self.ln_wo_grad(x)))  # this mul(gamma2) cannot be in-placed when FusedMLP is used
        # Cross attention
        x = x + self.drop_path(self.cross_attn(x, condition, freqs_cis=freqs_cis, freqs_cis_cross=freqs_cis_cross))
        return x


class AdaLNBeforeHead(nn.Module):
    def __init__(self, C, D, norm_layer):  # C: embed_dim, D: cond_dim
        super().__init__()
        self.C, self.D = C, D
        self.ln_wo_grad = norm_layer(C, elementwise_affine=False)

    def forward(self, x_BLC: torch.Tensor, cond_BD: torch.Tensor):
        return self.ln_wo_grad(x_BLC)
