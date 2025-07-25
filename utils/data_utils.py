import os
import json
from PIL import Image
from torch.utils.data import Dataset
from typing import Optional, Dict, Tuple
import torch
import numpy as np

class InatDataset(Dataset):
    
    def __init__(self, json_path: str, root_dir: str, transform: Optional[object] = None, class_to_idx: Optional[Dict[str, int]] = None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.class_to_idx = class_to_idx or {}
        
        self._load_data(json_path)
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
    
    def _load_data(self, json_path: str) -> None:
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load JSON file {json_path}: {e}")
        
        idx = max(self.class_to_idx.values(), default=-1) + 1
        
        for img in data['images']:
            file_name = img['file_name']
            class_name = file_name.split('/')[1].split('_', 1)[1]
            
            if class_name not in self.class_to_idx:
                self.class_to_idx[class_name] = idx
                idx += 1
            
            label = self.class_to_idx[class_name]
            path = os.path.join(self.root_dir, file_name)
            
            if os.path.exists(path):
                self.samples.append((path, label))
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        try:
            image_path, label = self.samples[idx]
            image = Image.open(image_path).convert("RGB")
            
            if self.transform:
                image = self.transform(image)
            
            return image, label
        except Exception as e:
            random_idx = np.random.randint(0, len(self.samples))
            return self.__getitem__(random_idx)