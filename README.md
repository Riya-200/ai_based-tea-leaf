# Tea Leaf Disease Screening & Classification

This repository contains the preprocessing, data loader, augmentation, and exploratory data analysis (EDA) pipeline for the **Kotagiri Tea Leaf Disease Dataset** (Original Data split).

## 📁 Workspace Directory Structure

```
ai_teleaf/
├── src/
│   ├── __init__.py
│   ├── dataset.py         # TeaLeafDataset loader: RGB conversion, resizing (224x224), normalization
│   └── augment.py         # DataAugmenter: On-the-fly flips, rotations, brightness/contrast scaling
├── eda_preprocessing.py   # Run full dataset preprocessing check, export metadata.csv & visualization grid
├── metadata.csv           # Detailed metadata dataframe of all 1,056 original images
├── eda_sample_grid.png    # Visualization sample grid of original vs. augmented tea leaf images
└── requirements.txt       # Project Python dependencies
```

## 📊 Dataset Statistics (Original Data Split)

- **Total Original Images**: `1,056`
- **Class Breakdown**:
  - `Disease` (Label 0): `606` images (Train: 480, Val: 80, Test: 46)
  - `Healthy` (Label 1): `450` images (Train: 360, Val: 70, Test: 20)
- **Input Dimension**: `(224, 224, 3)`
- **RGB Mean Stats**: `R: 0.4090, G: 0.4809, B: 0.2859`

## 🚀 Quickstart

Run the preprocessing and dataset audit verification:
```bash
python eda_preprocessing.py
```
