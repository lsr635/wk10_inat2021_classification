from dataclasses import dataclass
from typing import List, Optional
import torch
from pathlib import Path

@dataclass
class TrainingConfig:
    # Paths
    train_json_path: str = "/root/autodl-fs/Animals/data/split_jsons/full_150/train_60.json"
    val_json_path: str = "/root/autodl-fs/Animals/data/split_jsons/full_150/val_20.json"
    test_json_path: str = "/root/autodl-fs/Animals/data/split_jsons/full_150/test_20.json"
    root_dir: str = "/root/autodl-tmp/iNat21"
    model_save_dir: str = "./models"
    
    # Training params
    batch_size: int = 20
    num_epochs: int = 120
    patience: int = 20
    num_workers: int = 4
    
    # Model params
    dropout_rate: float = 0.6
    attention_ratio: int = 16
    
    # Optimizer params
    classifier_lr: float = 1e-4
    backbone_lr: float = 1e-5
    weight_decay: float = 1e-3
    
    # Loss params
    focal_alpha: float = 1.0
    focal_gamma: float = 2.0
    label_smoothing: float = 0.15
    
    # Augmentation params
    mixup_alpha: float = 0.2
    cutmix_alpha: float = 0.8
    
    # Unfreeze strategy
    unfreeze_epochs: List[int] = None
    
    def __post_init__(self):
        if self.unfreeze_epochs is None:
            self.unfreeze_epochs = [20, 40, 60]
        
        Path(self.model_save_dir).mkdir(parents=True, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")