import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def to01(x: torch.Tensor):
    # expects [-1,1] in [B,3,H,W] or [3,H,W]
    if x.ndim == 3: x = x.unsqueeze(0)
    return ((x.clamp(-1,1) + 1.0) * 0.5).clamp(0,1)

def psnr(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-10):
    # x,y: [-1,1], BxCxHxW or CxHxW
    x01, y01 = to01(x).float(), to01(y).float()
    mse = torch.mean((x01 - y01) ** 2)
    return (10.0 * torch.log10(1.0 / (mse + eps))).item()

def decode_final(var, idx_bl):
    """
    Return the final reconstruction image tensor [-1,1], shape [B,3,H,W].
    Uses last_one=True to save memory/time; falls back if API differs.
    """
    out = var.vae_proxy[0].idxBl_to_img(idx_bl, last_one=True, same_shape=True)
    if isinstance(out, list):
        img = out[-1]
    else:
        img = out
    return img

def to_pil_01(x: torch.Tensor) -> Image.Image:
    """
    x: [3,H,W] or [H,W] in [-1, 1]
    returns a PIL RGB image
    """
    x = x.detach().cpu()
    if x.ndim == 3:
        # [C,H,W] -> [H,W,C]
        if x.shape[0] == 1:
            x = x.expand(3, *x.shape[1:])
        assert x.shape[0] == 3, f"Expected 1 or 3 channels, got {x.shape[0]}"
        arr = x.permute(1, 2, 0).float().numpy()
    elif x.ndim == 2:
        arr = x.unsqueeze(-1).repeat(1,1,3).float().numpy()  # grayscale->RGB
    else:
        raise ValueError(f"Tensor must be [C,H,W] or [H,W], got shape {tuple(x.shape)}")

    arr = np.clip((arr + 1.0) * 0.5, 0.0, 1.0)  # [-1,1] -> [0,1]
    arr = (arr * 255.0 + 0.5).astype(np.uint8)
    return Image.fromarray(arr)

def save_scales_side_by_side(
    scales_list,
    out_path="scales_concat.png",
    batch_index=0,
    pad=8,
    font_size=16,
    caption_prefix="scale"
):
    """
    scales_list: list of tensors; each item is [B,3,H,W] or [3,H,W] in [-1,1]
    out_path: output image path
    batch_index: if tensors are batched, which sample to visualize
    pad: padding around and between tiles
    font_size: caption font size
    caption_prefix: text prefix, e.g., "scale"
    """
    # Normalize inputs to [3,H,W] tensors
    tiles = []
    for i, t in enumerate(scales_list):
        if not torch.is_tensor(t):
            raise ValueError(f"Item {i} is not a tensor")
        if t.ndim == 4:   # [B,3,H,W]
            if batch_index >= t.shape[0]:
                raise IndexError(f"batch_index {batch_index} out of range for item {i} with B={t.shape[0]}")
            t = t[batch_index]
        elif t.ndim == 3: # [3,H,W]
            pass
        else:
            raise ValueError(f"Item {i} must be [B,3,H,W] or [3,H,W], got {tuple(t.shape)}")
        tiles.append(to_pil_01(t))

    # assume same spatial size (you used same_shape=True)
    widths = [im.width for im in tiles]
    heights = [im.height for im in tiles]
    W = max(widths)
    H = max(heights)

    # set up font
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except:
        font = ImageFont.load_default()

    # measure caption height
    sample_text = f"{caption_prefix} 00"
    bbox = font.getbbox(sample_text)
    caption_h = bbox[3] - bbox[1] + 4

    n = len(tiles)
    canvas_w = pad + n*(W + pad)
    canvas_h = pad + H + 2 + caption_h + pad

    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    # paste tiles and captions
    x = pad
    for i, im in enumerate(tiles):
        # if any tile is smaller (shouldn't be), paste at top-left; otherwise they’re same size
        canvas.paste(im, (x, pad))
        caption = f"{caption_prefix} {i}"
        # center text under the tile
        tw = font.getlength(caption)
        tx = int(x + (W - tw) / 2)
        ty = pad + H + 2
        draw.text((tx, ty), caption, fill=(0,0,0), font=font)
        x += W + pad

    canvas.save(out_path)
    return out_path


def save_single_scale(
    scales_list,
    out_path="scales_concat.png",
    batch_index=0,
    pad=8,
    font_size=16,
    caption_prefix="scale"
):
    # Normalize inputs to [3,H,W] tensors
    tiles = []
    for i, t in enumerate(scales_list):
        if not torch.is_tensor(t):
            raise ValueError(f"Item {i} is not a tensor")
        if t.ndim == 4:   # [B,3,H,W]
            if batch_index >= t.shape[0]:
                raise IndexError(f"batch_index {batch_index} out of range for item {i} with B={t.shape[0]}")
            t = t[batch_index]
        elif t.ndim == 3: # [3,H,W]
            pass
        else:
            raise ValueError(f"Item {i} must be [B,3,H,W] or [3,H,W], got {tuple(t.shape)}")
        tiles.append(to_pil_01(t))

    # assume same spatial size (you used same_shape=True)
    widths = [im.width for im in tiles]
    heights = [im.height for im in tiles]
    W = max(widths)
    H = max(heights)

    tiles[0].save(out_path)
    return out_path