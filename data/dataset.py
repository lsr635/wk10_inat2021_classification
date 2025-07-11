# data/dataset.py
import os
import json
from PIL import Image
from torch.utils.data import Dataset

class InatDataset(Dataset):
    def __init__(self, json_path, root_dir, transform=None, class_to_idx=None):
        self.root_dir = root_dir
        self.transform = transform
        with open(json_path, 'r') as f:
            data = json.load(f)
        self.samples = []
        self.class_to_idx = class_to_idx or {}
        idx = max(self.class_to_idx.values(), default=-1) + 1
        for img in data['images']:
            file_name = img['file_name']
            class_name = file_name.split('/')[1].split('_', 1)[1]
            if class_name not in self.class_to_idx:
                self.class_to_idx[class_name] = idx
                idx += 1
            label = self.class_to_idx[class_name]
            path = os.path.join(root_dir, *file_name.split('/')[1:])
            self.samples.append((path, label))
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label