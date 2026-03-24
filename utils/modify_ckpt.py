import torch
import torch.nn as nn

def load_var(var_model: nn.Module, ckpt_path: str):
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

    # Remove resolution-specific buffers that should be recomputed such as attn bias and lvl1L
    state_dict.pop('attn_bias_for_masking', None)
    state_dict.pop('lvl_1L', None)
    state_dict.pop('pos_start', None)
    state_dict.pop('pos_1LC', None)

    m, u = var_model.load_state_dict(state_dict, strict=False)

    print(f"Unexpected keys: {u}")
    print(f"\nMissing keys: {m}")
