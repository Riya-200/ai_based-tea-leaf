import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.dataset import TeaLeafDataset
from src.augment import DataAugmenter

def run_eda_and_preprocessing():
    print("==================================================")
    print(" TEA LEAF DISEASE DATASET PREPROCESSING & EDA ")
    print("==================================================")

    # 1. Initialize dataset loader
    ds = TeaLeafDataset(normalize=False) # raw [0, 1] range for visualization
    print(f"\n[+] Total original samples collected: {len(ds.samples)}")

    # 2. Generate and export metadata DataFrame
    df_meta = ds.generate_metadata()
    csv_out = os.path.join(os.path.dirname(__file__), "metadata.csv")
    df_meta.to_csv(csv_out, index=False)
    print(f"[+] Exported dataset metadata to: {csv_out}")

    # 3. Print breakdown by split and class
    print("\n--- Split & Class Distribution ---")
    pivot = pd.crosstab(df_meta['split'], df_meta['class_name'], margins=True)
    print(pivot)

    # 4. RGB Channel Analysis
    print("\n--- RGB Channel Means (Scaled [0, 1]) ---")
    print(f"Red Channel Mean   : {df_meta['r_mean'].mean():.4f}")
    print(f"Green Channel Mean : {df_meta['g_mean'].mean():.4f}")
    print(f"Blue Channel Mean  : {df_meta['b_mean'].mean():.4f}")

    # 5. Test Preprocessed Array Loading
    print("\n--- Preprocessed Array Loading Test ---")
    X_train, y_train = ds.load_split_arrays('train')
    X_val, y_val = ds.load_split_arrays('val')
    X_test, y_test = ds.load_split_arrays('test')

    print(f"Train Set Shape : X={X_train.shape}, y={y_train.shape}")
    print(f"Val Set Shape   : X={X_val.shape}, y={y_val.shape}")
    print(f"Test Set Shape  : X={X_test.shape}, y={y_test.shape}")
    print(f"Pixel Range     : min={X_train.min():.2f}, max={X_train.max():.2f}")

    # 6. Generate Preprocessing & Augmentation Sample Visualization Grid
    print("\n--- Generating Visualization Grid ---")
    augmenter = DataAugmenter(rotation_deg=25, flip_prob=0.5)
    
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    fig.suptitle("Tea Leaf Disease: Preprocessed & Augmented Samples", fontsize=14, fontweight='bold')

    sample_disease = [s for s in ds.samples if s['class_name'] == 'Disease'][:2]
    sample_healthy = [s for s in ds.samples if s['class_name'] == 'Healthy'][:2]
    samples_to_plot = sample_disease + sample_healthy

    for idx, sample in enumerate(samples_to_plot):
        img_path = sample['filepath']
        with Image.open(img_path) as raw_img:
            img_rgb = raw_img.convert('RGB').resize((224, 224))
            
        aug_img = augmenter.augment_pil(img_rgb)
        
        # Plot Original
        ax_orig = axes[0, idx]
        ax_orig.imshow(img_rgb)
        ax_orig.set_title(f"Original ({sample['class_name']})", fontsize=10)
        ax_orig.axis('off')
        
        # Plot Augmented
        ax_aug = axes[1, idx]
        ax_aug.imshow(aug_img)
        ax_aug.set_title(f"Augmented ({sample['class_name']})", fontsize=10)
        ax_aug.axis('off')

    plt.tight_layout()
    grid_out = os.path.join(os.path.dirname(__file__), "eda_sample_grid.png")
    plt.savefig(grid_out, dpi=150)
    plt.close()
    print(f"[+] Saved preprocessing visualization grid to: {grid_out}")
    print("\n==================================================")
    print(" PREPROCESSING VERIFICATION COMPLETE SUCCESSFUL ")
    print("==================================================")

if __name__ == "__main__":
    run_eda_and_preprocessing()
