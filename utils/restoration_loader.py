import os

from PIL import Image
import numpy as np

from torch.utils.data import Dataset

import json
import torch
import random

DEGRADATIONS = {"Haze": 1, "Rain": 2, "Snow": 3, "Low-light": 4, "Motion blur": 5, "Mixed": 6, "toled": 7
                , "poled": 8}


class BaseDataset(Dataset):
    def __init__(self, json_path, transform, dino_transform=None):
        super(BaseDataset, self).__init__()

        self.json_path = json_path
        self.transform = transform
        self.dino_transform = dino_transform

        # Supported degradations
        self.valid_degradations = list(DEGRADATIONS.keys())
        self._init_ids()

    def get_dataset_stats(self, json_file):
        all_datasets = {}
        for id in json_file:
            ds = id['dataset']
            if ds not in all_datasets:
                all_datasets[ds] = []
            all_datasets[ds].append(id)

        nums = {}
        for ds in all_datasets:
            nums[ds] = len(all_datasets[ds])
            print(f"{ds}: {len(all_datasets[ds])}")
        return nums

    def _init_ids(self):
        self.ids = []
        self.sample_classes = []
        loaded_json = json.load(open(self.json_path))

        new_lines = []
        for line in loaded_json:
            new_lines.append(line)
        loaded_json = new_lines.copy()

        # Count occurrences of each dataset
        self.counts = self.get_dataset_stats(loaded_json)

        for line in loaded_json:
            self.ids.append(line)
            self.sample_classes.append(DEGRADATIONS[line['degradation']])

        print(f"Initialized {len(self.ids)} IDs")

    def __getitem__(self, idx):
        sample = self.ids[idx]
        degraded_name, clean_name = sample['image_path'], sample['target_path']

        ds = sample["dataset"]
        if clean_name is None or not os.path.exists(clean_name):
            raise FileNotFoundError(
                f"Image path {clean_name} is invalid or does not exist.")

        if degraded_name is None or not os.path.exists(degraded_name):
            raise FileNotFoundError(
                f"Image path {degraded_name} is invalid or does not exist.")

        degrad_img = Image.open(degraded_name).convert('RGB')
        clean_img = Image.open(clean_name).convert('RGB')

        # Apply paired transformations
        clean_img, degrad_img = self.transform(clean_img, degrad_img)

        return clean_img, degrad_img, 1000, ds, degraded_name  # Input to VAR, degraded image as condition, label fixed to 1000, dataset

    def __len__(self):
        return len(self.ids)