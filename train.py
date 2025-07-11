import os
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import InatDataset
from models.resnet_transfer import ResNet50Transfer
from utils.transforms import train_transform, val_transform, tta_transforms
from utils.mixup import mixup_data, mixup_criterion

# ----- 配置 -----
TRAIN_JSON = "/root/autodl-fs/Animals/train_mini_subset_00000_01000.json"
VAL_JSON = "/root/autodl-fs/Animals/val_subset_00000_01000.json"
TRAIN_DIR = "/root/autodl-fs/Animals/train_mini"
VAL_DIR = "/root/autodl-fs/Animals/val"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_EPOCHS = 100
BATCH_SIZE = 64
PATIENCE = 8
NUM_WORKERS = 4

def evaluate_tta(model, dataset, criterion, tta_transforms):
    model.eval()
    val_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc="Evaluating"):
            path, label = dataset.samples[i]
            image = Image.open(path).convert("RGB")
            preds = []
            for t in tta_transforms:
                img = t(image).unsqueeze(0).to(DEVICE)
                out = model(img)
                preds.append(torch.softmax(out, dim=1))
            pred = torch.mean(torch.stack(preds), dim=0)
            loss = criterion(pred, torch.tensor([label], dtype=torch.long).to(DEVICE))
            val_loss += loss.item()
            pred_class = pred.argmax(dim=1).item()
            correct += int(pred_class == label)
            total += 1
    return val_loss / total, 100 * correct / total

def train():
    train_dataset = InatDataset(TRAIN_JSON, TRAIN_DIR, transform=train_transform)
    val_dataset = InatDataset(VAL_JSON, VAL_DIR, transform=val_transform, class_to_idx=train_dataset.class_to_idx)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model = ResNet50Transfer(num_classes=len(train_dataset.class_to_idx)).to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    best_val_acc, early_stop_counter = 0.0, 0
    train_losses, train_accuracies, val_losses, val_accuracies = [], [], [], []

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            images, targets_a, targets_b, lam = mixup_data(images, labels, alpha=0.4)
            outputs = model(images)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (lam * predicted.eq(targets_a).sum().item() +
                        (1 - lam) * predicted.eq(targets_b).sum().item())

        train_acc = 100 * correct / total
        val_loss, val_acc = evaluate_tta(model, val_dataset, criterion, tta_transforms)

        print(f"Epoch {epoch+1}: Train Loss = {running_loss:.2f}, Train Acc = {train_acc:.2f}%")
        print(f"               Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.2f}%")

        train_losses.append(running_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            early_stop_counter = 0
            torch.save(model.state_dict(), "best_model_val_5.pth")
            print("Model saved.")
        else:
            early_stop_counter += 1
            if early_stop_counter >= PATIENCE:
                print("Early stopping triggered.")
                break

        pd.DataFrame({
            'epoch': list(range(1, epoch + 2)),
            'train_loss': train_losses,
            'train_acc': train_accuracies,
            'val_loss': val_losses,
            'val_acc': val_accuracies
        }).to_csv("training_metrics_5.csv", index=False)

        scheduler.step()

if __name__ == "__main__":
    train()