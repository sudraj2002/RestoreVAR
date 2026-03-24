import os
from pathlib import Path
from PIL import Image
import shutil

src_root = Path("results_gen_split")
dst_root = Path("results_for_metric/")

# Walk through the source directory
for root, dirs, files in os.walk(src_root):
    rel_path = Path(root).relative_to(src_root)
    dst_dir = dst_root / rel_path
    dst_dir.mkdir(parents=True, exist_ok=True)

    for file in files:
        if 'CDD_haze_rain' in str(dst_dir) or 'LOLBlur' in str(dst_dir):
            src_file = Path(root) / file
            dst_file = dst_dir / file
            shutil.copyfile(src_file, dst_file)
            continue

        src_file = Path(root) / file
        try:
            img = Image.open(src_file).convert("RGB")  # ensure RGB
            dst_file = dst_dir / (src_file.stem + ".png")
            img.save(dst_file, "PNG")
        except Exception as e:
            print(f"Skipping {src_file}: {e}")


