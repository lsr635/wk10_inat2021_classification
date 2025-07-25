import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import json
import pandas as pd
import numpy as np
import time
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix

from models import ResNet50Transfer
from utils import TrainingConfig

MODEL_PATH = "./best_model_val_6.pth"
TEST_JSON_PATH = "/root/autodl-fs/Animals/data/split_jsons/full_150/test_20.json"
ROOT_DIR = "/root/autodl-tmp/iNat21"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    
    class_to_idx = checkpoint['class_to_idx']
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    num_classes = len(class_to_idx)
    
    model = ResNet50Transfer(num_classes=num_classes, dropout_rate=0.6)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    
    print(f"Model loaded with {num_classes} classes")
    return model, class_to_idx, idx_to_class

def main():
    model, class_to_idx, idx_to_class = load_model()
    print("Testing complete!")

if __name__ == "__main__":
    main()