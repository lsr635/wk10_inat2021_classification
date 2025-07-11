# wk10_inat2021_classification
## Deep learning practice


## iNat2021 Animal Classification (1001 Classes)

This project implements a deep learning pipeline for fine-grained image classification on the [iNaturalist 2021](https://www.kaggle.com/competitions/inaturalist-2021-fgvc8) dataset, covering 1001 animal species. The model is trained using PyTorch with transfer learning (ResNet-50), and supports Mixup augmentation and inference on an unlabeled test set.

---
## Features

- 1001-way classification using ResNet-50
- Mixup data augmentation
- Custom training/validation dataloader
- Inference with confidence output
- Confidence histogram & pseudo-label generation

---
##  Training

```bash
python train.py
```

You can customize parameters such as batch size, learning rate, and epochs inside the script.

---

## Sample Output

| true_label                                                                 | predicted_label                                                              | confidence | correct |
|----------------------------------------------------------------------------|-------------------------------------------------------------------------------|------------|---------|
| Animalia_Arthropoda_Insecta_Lepidoptera_Erebidae_Arctia_virginalis        | Animalia_Arthropoda_Insecta_Coleoptera_Lampyridae_Photinus_pyralis           | 0.6199     | FALSE   |
| Animalia_Arthropoda_Insecta_Lepidoptera_Erebidae_Arctia_virginalis        | Animalia_Arthropoda_Insecta_Lepidoptera_Erebidae_Arctia_virginalis           | 0.9329     | TRUE    |
| Animalia_Arthropoda_Insecta_Lepidoptera_Erebidae_Arctia_virginalis        | Animalia_Arthropoda_Insecta_Diptera_Tachinidae_Hystricia_abrupta             | 0.0956     | FALSE   |
| Animalia_Arthropoda_Insecta_Lepidoptera_Erebidae_Arctia_virginalis        | Animalia_Arthropoda_Insecta_Lepidoptera_Erebidae_Arctia_virginalis           | 0.7227     | TRUE    |
| Animalia_Arthropoda_Insecta_Lepidoptera_Erebidae_Arctia_virginalis        | Animalia_Arthropoda_Insecta_Lepidoptera_Erebidae_Arctia_virginalis           | 0.0834     | TRUE    |
| Animalia_Arthropoda_Insecta_Lepidoptera_Erebidae_Arctia_virginalis        | Animalia_Arthropoda_Insecta_Lepidoptera_Erebidae_Arctia_virginalis           | 0.5258     | TRUE    |
| Animalia_Arthropoda_Insecta_Lepidoptera_Erebidae_Arctia_virginalis        | Animalia_Arthropoda_Insecta_Lepidoptera_Erebidae_Arctia_virginalis           | 0.7871     | TRUE    |

---

## Resources

- Dataset format adapted from: [iNaturalist 2021](https://www.kaggle.com/competitions/inaturalist-2021-fgvc8)
- Base model: `torchvision.models.resnet50` with custom classifier

---

