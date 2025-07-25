import os
import logging
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import TrainingConfig
from .data_utils import InatDataset
from .transforms import TransformFactory
from .augmentation import DataAugmentation
from models import ResNet50Transfer, FocalLoss

class Evaluator:
    
    def __init__(self, model: nn.Module, device: torch.device, tta_transforms: list):
        self.model = model
        self.device = device
        self.tta_transforms = tta_transforms
    
    def evaluate_with_tta(self, dataset: InatDataset, criterion: nn.Module) -> tuple:
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for i in tqdm(range(len(dataset)), desc="Evaluating"):
                path, label = dataset.samples[i]
                image = Image.open(path).convert("RGB")
                preds = []
                
                for j, transform in enumerate(self.tta_transforms):
                    if j == 4:  # FiveCrop transform
                        imgs = transform(image)
                        for img in imgs:
                            img = img.unsqueeze(0).to(self.device)
                            out = self.model(img)
                            preds.append(torch.softmax(out, dim=1))
                    else:
                        img = transform(image).unsqueeze(0).to(self.device)
                        out = self.model(img)
                        preds.append(torch.softmax(out, dim=1))
                
                pred = torch.mean(torch.stack(preds), dim=0)
                loss = criterion(pred, torch.tensor([label], dtype=torch.long).to(self.device))
                val_loss += loss.item()
                pred_class = pred.argmax(dim=1).item()
                if pred_class == label:
                    correct += 1
                total += 1
        
        return val_loss / total, 100 * correct / total

class Trainer:
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.logger = self._setup_logger()
        
        self._setup_data()
        self._setup_model()
        self._setup_optimizer()
        self._setup_evaluator()
        
        self.best_val_acc = 0.0
        self.early_stop_counter = 0
        self.train_history = {
            'epoch': [], 'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }
    
    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger("trainer")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _setup_data(self) -> None:
        train_transform = TransformFactory.get_train_transform()
        val_transform = TransformFactory.get_val_transform()
        
        self.train_dataset = InatDataset(self.config.train_json_path, self.config.root_dir, train_transform)
        self.val_dataset = InatDataset(self.config.val_json_path, self.config.root_dir, val_transform, self.train_dataset.class_to_idx)
        
        self.train_loader = DataLoader(
            self.train_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True, 
            num_workers=self.config.num_workers, 
            pin_memory=True
        )
        
        self.logger.info(f"Training samples: {len(self.train_dataset)}")
        self.logger.info(f"Validation samples: {len(self.val_dataset)}")
        self.logger.info(f"Number of classes: {len(self.train_dataset.class_to_idx)}")
    
    def _setup_model(self) -> None:
        num_classes = len(self.train_dataset.class_to_idx)
        self.model = ResNet50Transfer(
            num_classes=num_classes,
            dropout_rate=self.config.dropout_rate,
            attention_ratio=self.config.attention_ratio
        ).to(self.config.device)
        
        self.criterion = FocalLoss(
            alpha=self.config.focal_alpha,
            gamma=self.config.focal_gamma,
            label_smoothing=self.config.label_smoothing
        )
        
        self.logger.info(f"Model created with {num_classes} classes")
    
    def _setup_optimizer(self) -> None:
        classifier_params = list(self.model.classifier.parameters()) + list(self.model.attention.parameters())
        backbone_params = [p for n, p in self.model.resnet50.named_parameters() if p.requires_grad]
        
        self.optimizer = torch.optim.AdamW([
            {'params': classifier_params, 'lr': self.config.classifier_lr},
            {'params': backbone_params, 'lr': self.config.backbone_lr}
        ], weight_decay=self.config.weight_decay, betas=(0.9, 0.999))
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=20, T_mult=2, eta_min=1e-7
        )
    
    def _setup_evaluator(self) -> None:
        tta_transforms = TransformFactory.get_tta_transforms()
        self.evaluator = Evaluator(self.model, self.config.device, tta_transforms)
    
    def _unfreeze_layers(self, epoch: int) -> bool:
        unfreeze_map = {
            self.config.unfreeze_epochs[0]: "layer4",
            self.config.unfreeze_epochs[1]: "layer3", 
            self.config.unfreeze_epochs[2]: "layer2"
        }
        
        if epoch in unfreeze_map:
            layer_name = unfreeze_map[epoch]
            self.model.unfreeze_layer(layer_name)
            self.logger.info(f"Unfroze {layer_name}")
            return True
        return False
    
    def _train_epoch(self, epoch: int) -> tuple:
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        aug_prob = max(0.4, 0.8 - epoch / 60)
        
        for images, labels in tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}"):
            images, labels = images.to(self.config.device, non_blocking=True), labels.to(self.config.device, non_blocking=True)
            
            aug_choice = np.random.choice(['mixup', 'cutmix', 'none'], p=[0.35, 0.35, 0.3])
            
            if aug_choice == 'mixup' and np.random.random() < aug_prob:
                images, targets_a, targets_b, lam = DataAugmentation.mixup_data(images, labels, self.config.mixup_alpha)
                outputs = self.model(images)
                loss = DataAugmentation.mixed_criterion(self.criterion, outputs, targets_a, targets_b, lam)
            elif aug_choice == 'cutmix' and np.random.random() < aug_prob:
                images, targets_a, targets_b, lam = DataAugmentation.cutmix_data(images, labels, self.config.cutmix_alpha)
                outputs = self.model(images)
                loss = DataAugmentation.mixed_criterion(self.criterion, outputs, targets_a, targets_b, lam)
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                targets_a, targets_b, lam = labels, labels, 1.0
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            
            if aug_choice in ['mixup', 'cutmix'] and np.random.random() < aug_prob:
                correct += (lam * predicted.eq(targets_a).sum().item()
                           + (1 - lam) * predicted.eq(targets_b).sum().item())
            else:
                correct += predicted.eq(labels).sum().item()
        
        train_loss = running_loss / len(self.train_loader)
        train_acc = 100 * correct / total
        
        return train_loss, train_acc
    
    def _save_checkpoint(self, epoch: int, val_acc: float) -> None:
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_acc': val_acc,
            'class_to_idx': self.train_dataset.class_to_idx,
            'config': self.config
        }
        
        model_path = os.path.join(self.config.model_save_dir, "best_model.pth")
        torch.save(checkpoint, model_path)
        self.logger.info(f"Model saved with val_acc: {val_acc:.2f}%")
    
    def _save_training_history(self) -> None:
        df = pd.DataFrame(self.train_history)
        history_path = "training_metrics.csv"
        df.to_csv(history_path, index=False)
    
    def train(self) -> None:
        self.logger.info("Starting training...")
        
        for epoch in range(self.config.num_epochs):
            if self._unfreeze_layers(epoch):
                self._setup_optimizer()
            
            train_loss, train_acc = self._train_epoch(epoch)
            val_loss, val_acc = self.evaluator.evaluate_with_tta(self.val_dataset, self.criterion)
            
            self.train_history['epoch'].append(epoch + 1)
            self.train_history['train_loss'].append(train_loss)
            self.train_history['train_acc'].append(train_acc)
            self.train_history['val_loss'].append(val_loss)
            self.train_history['val_acc'].append(val_acc)
            
            current_lr = self.optimizer.param_groups[0]['lr']
            self.logger.info(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Train Acc = {train_acc:.2f}%")
            self.logger.info(f"               Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.2f}%")
            self.logger.info(f"               LR = {current_lr:.6f}")
            
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.early_stop_counter = 0
                self._save_checkpoint(epoch, val_acc)
            else:
                self.early_stop_counter += 1
                if self.early_stop_counter >= self.config.patience:
                    self.logger.info("Early stopping triggered.")
                    break
            
            self._save_training_history()
        
        self.logger.info(f"Training completed. Best validation accuracy: {self.best_val_acc:.2f}%")