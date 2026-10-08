import random
from PIL import Image, ImageEnhance
import numpy as np

class DataAugmenter:
    """
    On-the-fly Data Augmentation pipeline for Tea Leaf Disease images.
    Applies geometric and color transformations to expand dataset diversity.
    """
    def __init__(self, rotation_deg=30, flip_prob=0.5, brightness_range=(0.8, 1.2), contrast_range=(0.8, 1.2)):
        self.rotation_deg = rotation_deg
        self.flip_prob = flip_prob
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range

    def augment_pil(self, pil_img):
        """Applies random transformations to a PIL Image."""
        img = pil_img.copy()

        # 1. Random Horizontal Flip
        if random.random() < self.flip_prob:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        # 2. Random Vertical Flip
        if random.random() < self.flip_prob:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)

        # 3. Random Rotation
        if self.rotation_deg > 0:
            angle = random.uniform(-self.rotation_deg, self.rotation_deg)
            img = img.rotate(angle, resample=Image.BILINEAR)

        # 4. Random Brightness Adjustment
        if self.brightness_range:
            factor = random.uniform(*self.brightness_range)
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(factor)

        # 5. Random Contrast Adjustment
        if self.contrast_range:
            factor = random.uniform(*self.contrast_range)
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(factor)

        return img

    def augment_and_normalize(self, filepath, target_size=(224, 224), mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        """Loads image, applies augmentation, resizes, and normalizes to float32 tensor array."""
        with Image.open(filepath) as img:
            img_rgb = img.convert('RGB')
            if img_rgb.size != target_size:
                img_rgb = img_rgb.resize(target_size, Image.BILINEAR)
                
            aug_img = self.augment_pil(img_rgb)
            img_arr = np.array(aug_img, dtype=np.float32) / 255.0
            
            # Apply standardization
            mean_arr = np.array(mean, dtype=np.float32)
            std_arr = np.array(std, dtype=np.float32)
            img_norm = (img_arr - mean_arr) / std_arr
            
            return img_norm
