import argparse

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--calc_metrics",
        action="store_true",
        help="Whether to compute PSNR, SSIM, LPIPS, and FID metrics."
    )
    parser.add_argument(
        "--model_depth",
        type=int,
        choices=[16, 20, 24, 30],
        required=True,
        help="Depth of the VAR model (16, 20, 24, or 30)."
    )
    parser.add_argument(
        "--json_path",
        type=str,
        required=True,
        help="Path to the JSON file specifying the evaluation dataset."
    )
    parser.add_argument(
        "--patch_nums",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5, 6, 8, 10, 13, 16],
        help="Patch numbers for multi-scale tokenization."
    )
    parser.add_argument(
        "--refiner",
        action='store_true',
        help="Use refiner."
    )
    parser.add_argument(
        "--vae_ckpt",
        type=str,
        required=True,
        default='ckpts/vae_ch160v4096z32.pth',
        help="Path to the VQ-VAE checkpoint file."
    )
    parser.add_argument(
        "--var_ckpt",
        type=str,
        required=True,
        default='ckpts/var_d16.pth',
        help="Path to the VAR model checkpoint file."
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        default="",
        help="Directory to save the output images."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Optional seed for reproducibility."
    )

    args = parser.parse_args()
    return args
