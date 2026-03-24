import os
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import pyiqa
import random

parent_dir = 'results_for_metric'
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create metrics
musiq = pyiqa.create_metric('musiq', device=device)
clipiqa = pyiqa.create_metric('clipiqa', device=device)

preprocess = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])


# List all subdirectories
subdirs = sorted([os.path.join(parent_dir, d) for d in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, d))])

for subdir in subdirs:
    image_files = sorted([
        os.path.join(subdir, f) for f in os.listdir(subdir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])
    random.shuffle(image_files)

    if len(image_files) == 0:
        print(f"{os.path.basename(subdir):<30} | {'N/A':>8} | {'N/A':>8} | {'N/A':>8}")
        continue

    musiq_scores, clipiqa_scores = [], []
    counter = 0
    for fpath in image_files:
        try:
            with torch.no_grad():
                musiq_scores.append(musiq(fpath).item())
                clipiqa_scores.append(clipiqa(fpath).item())
        except Exception as e:
            print(f"Error on {fpath}: {e}")
            continue

    mean_musiq = sum(musiq_scores) / len(musiq_scores)
    mean_clipiqa = sum(clipiqa_scores) / len(clipiqa_scores)

    print(f"{os.path.basename(subdir):<30} | {mean_musiq:8.4f} | {mean_clipiqa:8.4f}")