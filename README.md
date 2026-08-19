# Waste Type Identification — Group 26

Machine Learning project for automatic classification of waste images into eight categories. The final model is **EfficientNet-B0** fine-tuned with **weighted Cross-Entropy loss** and validated through **5-fold stratified cross-validation**.

## Team

| Name | Student ID |
|------|------------|
| Francesco Vittorio Allocca
| Aniello Di Meglio
| Gennaro Foschillo

## Problem

Given an RGB image of a waste item, predict one of the following classes:

| Label | Class |
|-------|-------|
| 0 | Battery |
| 1 | Clothing |
| 2 | Glass |
| 3 | Metal |
| 4 | Organic |
| 5 | Papery |
| 6 | Plastic |
| 7 | Undifferentiated |

The main evaluation metric is **Balanced Accuracy**, which accounts for class imbalance in the dataset.

## Dataset

The labeled dataset is stored in `waste_type_identification/` and contains **15,515** JPEG images organized by class folders:

| Class | Samples |
|-------|---------|
| Battery | 945 |
| Clothing | 7,302 |
| Glass | 2,011 |
| Metal | 769 |
| Organic | 985 |
| Papery | 1,941 |
| Plastic | 865 |
| Undifferentiated | 697 |

Cross-validation fold assignments are saved in `split.csv` (`relative_path`, `label`, `class_name`, `cv_fold`, `split`).

## Approach

- **Architecture:** EfficientNet-B0 with ImageNet pretrained weights (`IMAGENET1K_V1`)
- **Training strategy:** Full fine-tuning of all layers
- **Loss:** Weighted Cross-Entropy (inverse-frequency class weights recomputed per fold)
- **Validation:** Stratified 5-fold cross-validation
- **Model selection:** The fold whose validation Balanced Accuracy is closest to the CV mean is selected as the representative model
- **Preprocessing:** Resize to 224×224, ImageNet normalization, no data augmentation
- **Optimizer:** AdamW (lr = 1e-4, weight decay = 1e-4)
- **Early stopping:** Patience = 10 epochs (max 30 epochs)
- **Reproducibility:** Fixed seed (`SEED = 42`)

## Project Structure

```text
Group_26/
├── waste_type_identification/                             # Labeled image dataset
├── efficientnet_b0_weighted_ce_5fold_colab.ipynb          # Training & cross-validation (Google Colab)
├── efficientnet_b0_weighted_ce_5fold_selected_model.pt    # Best model checkpoint
├── test_script_submission.ipynb                           # Evaluation/submission notebook
├── split.csv                                              # CV fold assignments
├── predictions.npy                                        # Generated predictions
├── Report_Group26.pdf                                     # Project report
├── Group26Members.txt                                     # Team members
└── slide.txt                                              # Presentation link

```

## Requirements

Training is designed to run on **Google Colab** with GPU support. Main dependencies:

* Python 3.x
* PyTorch & torchvision
* scikit-learn
* pandas, numpy
* matplotlib, tqdm
* Pillow

## Usage

### 1. Training (Google Colab)

Open `efficientnet_b0_weighted_ce_5fold_colab.ipynb` in Colab and:

1. Mount Google Drive.
2. Update the paths in **Section 1** to point to your dataset ZIP (`waste_type_identification.zip`) and project folder.
3. Run all cells.

The notebook will:

* Extract the dataset locally
* Run 5-fold cross-validation
* Save logs, confusion matrices, and the best checkpoint

Expected outputs (on Google Drive):

* `logs/efficientnet_b0_weighted_ce_5fold/` — JSON summary and confusion matrix CSV
* `figures/efficientnet_b0_weighted_ce_5fold/` — Normalized confusion matrix plot

### 2. Inference & Submission

Use `test_script_submission.ipynb` for local or Colab evaluation:

1. Place the notebook and `efficientnet_b0_weighted_ce_5fold_selected_model.pt` in the same directory.
2. Create an `eval/` folder with test images.
3. Run all cells.

The notebook implements:

* **`load_model()`** — Loads EfficientNet-B0 and the saved checkpoint
* **`predict(model, X)`** — Preprocesses uint8 batches `(batch, H, W, 3)` and returns predicted labels `(batch, 1)`

Predictions are saved to `predictions.npy`.

### Example: loading the model locally

```python
import torch
from torch import nn
from torchvision import models

NUM_CLASSES = 8
MODEL_PATH = "./efficientnet_b0_weighted_ce_5fold_selected_model.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    return model

```

## Deliverables

| File | Description |
| --- | --- |
| `efficientnet_b0_weighted_ce_5fold_colab.ipynb` | Full training pipeline with 5-fold CV |
| `efficientnet_b0_weighted_ce_5fold_selected_model.pt` | Selected model weights |
| `test_script_submission.ipynb` | Inference script for evaluation |
| `split.csv` | Train/validation split per fold |
| `Report_Group26.pdf` | Written project report |
| `Presentation slides` | Canva presentation |

## Notes

* **Section 2** of `test_script_submission.ipynb` must not be modified — it replicates the official evaluation code.
* Class weights are computed independently for each fold using only the training subset, so validation samples do not leak into the loss.
* The dataset is heavily imbalanced (Clothing is the majority class); weighted Cross-Entropy helps mitigate this during training.

## License

Academic project — Machine Learning course, Group 26.
