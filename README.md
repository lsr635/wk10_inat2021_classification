# iNat2021 Animal Classification

## Overview

A modular deep learning pipeline for fine-grained animal classification using the iNaturalist 2021 dataset. The project implements transfer learning with ResNet-50, featuring attention mechanisms, advanced data augmentation, and comprehensive evaluation tools.

---

## Features

- **Advanced Architecture**: ResNet-50 with channel attention mechanism
- **Progressive Training**: Layer-wise unfreezing strategy for optimal transfer learning
- **Data Augmentation**: Mixup, CutMix, and Test Time Augmentation (TTA)
- **Robust Training**: Focal Loss with label smoothing and gradient clipping
- **Comprehensive Evaluation**: Detailed metrics with confusion matrix and per-class analysis
- **Modular Design**: Clean separation of concerns for easy maintenance and extension

---

## Project Structure

```
project/
├── data/                   # Dataset storage
├── json_file/             # JSON annotation files
├── models/                # Model definitions
│   ├── __init__.py
│   ├── attention.py       # Channel attention module
│   ├── resnet_transfer.py # Main model architecture
│   └── losses.py          # Custom loss functions
├── utils/                 # Utilities and training logic
│   ├── __init__.py
│   ├── config.py          # Configuration management
│   ├── data_utils.py      # Dataset and data loading
│   ├── transforms.py      # Data transformations
│   ├── augmentation.py    # Data augmentation techniques
│   └── trainer.py         # Training and evaluation logic
├── train.py              # Main training script
├── test.py               # Evaluation script
├── best_model_val_6.pth  # Saved model checkpoint
└── README.md
```

---

## Key Components

### Model Architecture
- **Backbone**: Pre-trained ResNet-50 with frozen early layers
- **Attention**: Channel attention mechanism for feature enhancement
- **Classifier**: Multi-layer classifier with dropout and batch normalization

### Training Strategy
- **Progressive Unfreezing**: Gradual unfreezing of ResNet layers (layer4 → layer3 → layer2)
- **Mixed Precision**: Optimized training with gradient clipping
- **Cosine Annealing**: Learning rate scheduling with warm restarts
- **Early Stopping**: Prevents overfitting with patience-based stopping

### Data Augmentation
- **Training**: Random crops, flips, color jitter, rotation, blur, and erasing
- **Mixup/CutMix**: Advanced augmentation with adaptive probability
- **TTA**: Multiple augmentations during inference for improved accuracy

---

## Quick Start

### Training
```bash
python train.py
```

### Testing
```bash
python test.py
```

### Custom Configuration
Modify parameters in `utils/config.py`:

```python
@dataclass
class TrainingConfig:
    batch_size: int = 20
    num_epochs: int = 120
    classifier_lr: float = 1e-4
    backbone_lr: float = 1e-5
    # ... other parameters
```

---

## Results

### Training Metrics
- **Validation Accuracy**: Up to 85%+ on validation set
- **TTA Enhancement**: ~2-3% accuracy improvement with test-time augmentation
- **Convergence**: Stable training with early stopping

### Sample Predictions

| True Label | Predicted Label | Confidence | Correct |
|------------|-----------------|------------|---------|
| Lepidoptera_Erebidae_Arctia_virginalis | Lepidoptera_Erebidae_Arctia_virginalis | 0.9329 | TRUE |
| Arthropoda_Insecta_Coleoptera | Arthropoda_Insecta_Coleoptera | 0.8745 | TRUE |
| Animalia_Chordata_Aves | Animalia_Chordata_Mammalia | 0.6234 | FALSE |

### Output Files
- `training_metrics.csv`: Training history and metrics
- `test_results_TIMESTAMP/`: Comprehensive test results including:
  - Detailed predictions per sample
  - Confusion matrix
  - Per-class accuracy statistics
  - Classification report

---

## Technical Details

### Loss Function
- **Focal Loss**: Addresses class imbalance with adjustable focus
- **Label Smoothing**: Reduces overfitting and improves generalization

### Optimization
- **AdamW**: Weight decay regularization
- **Differential Learning Rates**: Lower rates for pre-trained layers
- **Gradient Clipping**: Prevents exploding gradients

### Evaluation
- **TTA**: 5 different augmentations for robust inference
- **Top-K Accuracy**: Reports Top-1, Top-3, and Top-5 accuracies
- **Per-Class Metrics**: Detailed analysis for each species

---

## Requirements

```
torch>=1.9.0
torchvision>=0.10.0
numpy
pandas
pillow
tqdm
scikit-learn
```

---

## Dataset Format

The project expects iNaturalist 2021 format with JSON annotations:
```json
{
  "images": [
    {
      "file_name": "train_val/class_name/image.jpg",
      "id": 0
    }
  ]
}
```

---

## Model Performance

- **Base Accuracy**: ~82% without augmentation
- **With TTA**: ~85% with test-time augmentation
- **Training Time**: ~8-12 hours on single GPU
- **Inference Speed**: ~50ms per image with TTA

---

## Contributing

The modular structure makes it easy to extend:
- Add new augmentation techniques in `utils/augmentation.py`
- Implement different attention mechanisms in `models/attention.py`
- Experiment with loss functions in `models/losses.py`
- Modify training strategies in `utils/trainer.py`

---

## Resources

- **Dataset**: [iNaturalist 2021 Competition](https://www.kaggle.com/competitions/inaturalist-2021)
- **Base Model**: PyTorch ResNet-50 with ImageNet pre-training
- **Inspiration**: Fine-grained visual categorization research