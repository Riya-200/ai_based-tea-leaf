import os
from PIL import Image
import numpy as np
import pandas as pd

class TeaLeafDataset:
    """
    Dataset handler for Tea Leaf Disease Screening (Original Data split).
    Handles format validation, RGB conversion, resizing, and normalization.
    """
    CLASS_MAP = {'Disease': 0, 'Healthy': 1}
    INV_CLASS_MAP = {0: 'Disease', 1: 'Healthy'}

    def __init__(self, base_dir=None, target_size=(224, 224), normalize=True):
        if base_dir is None:
            self.base_dir = r"C:\Users\lenovo\Desktop\ai_teleaf\K-Kotagiri Tea Leaf Dataset\New folder\Kotagiri Tea Leaf Dataset\Original Data"
        else:
            self.base_dir = base_dir
            
        self.target_size = target_size
        self.normalize = normalize
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self.samples = self._collect_samples()

    def _collect_samples(self):
        """Scans the dataset directory and gathers paths, splits, and labels."""
        samples = []
        splits = {'Train': 'train', 'valid': 'val', 'Test': 'test'}
        
        for split_folder, split_name in splits.items():
            split_path = os.path.join(self.base_dir, split_folder)
            if not os.path.exists(split_path):
                continue
                
            for class_name in ['Disease', 'Healthy']:
                class_path = os.path.join(split_path, class_name)
                if not os.path.exists(class_path):
                    continue
                    
                for fname in os.listdir(class_path):
                    if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                        full_path = os.path.join(class_path, fname)
                        samples.append({
                            'filepath': full_path,
                            'filename': fname,
                            'split': split_name,
                            'class_name': class_name,
                            'label': self.CLASS_MAP[class_name]
                        })
        return samples

    def load_image(self, filepath, apply_norm=None):
        """
        Loads an image, converts to RGB, resizes to target_size,
        and optionally normalizes to float32 [0, 1] / ImageNet standardized tensor.
        """
        if apply_norm is None:
            apply_norm = self.normalize

        with Image.open(filepath) as img:
            # Enforce RGB color space
            img_rgb = img.convert('RGB')
            # Resize if different from target_size
            if img_rgb.size != self.target_size:
                img_rgb = img_rgb.resize(self.target_size, Image.BILINEAR)
                
            img_array = np.array(img_rgb, dtype=np.float32) / 255.0
            
            if apply_norm:
                img_array = (img_array - self.mean) / self.std
                
            return img_array

    def generate_metadata(self):
        """Generates a DataFrame with image stats (mean RGB, resolution, etc.)."""
        meta_list = []
        for sample in self.samples:
            img = self.load_image(sample['filepath'], apply_norm=False)
            r_mean, g_mean, b_mean = img[:, :, 0].mean(), img[:, :, 1].mean(), img[:, :, 2].mean()
            
            item = sample.copy()
            item['height'] = self.target_size[1]
            item['width'] = self.target_size[0]
            item['r_mean'] = round(float(r_mean), 4)
            item['g_mean'] = round(float(g_mean), 4)
            item['b_mean'] = round(float(b_mean), 4)
            meta_list.append(item)
            
        return pd.DataFrame(meta_list)

    def load_split_arrays(self, split='train'):
        """Loads all preprocessed image arrays and labels for a given split (train/val/test)."""
        split_samples = [s for s in self.samples if s['split'] == split]
        X, y = [], []
        for sample in split_samples:
            img_arr = self.load_image(sample['filepath'])
            X.append(img_arr)
            y.append(sample['label'])
            
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)
