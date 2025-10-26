""" filename: datasets.py """

import os
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset


# ============================================================================
# Base Dataset Classes
# ============================================================================

class BaseDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, mask_transform=None):
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.mask_transform = mask_transform
        self.samples = []
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        raise NotImplementedError
    
    def _load_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image
    
    def _load_mask(self, mask_path):
        if mask_path is None or not Path(mask_path).exists():
            return None
        mask = Image.open(mask_path).convert('L')
        if self.mask_transform:
            mask = self.mask_transform(mask)
        return mask


class AnomalyDataset(BaseDataset):
    def __init__(self, root_dir, category, split='train', transform=None, mask_transform=None):
        super().__init__(root_dir, split, transform, mask_transform)
        self.category = category
        self._load_samples()
    
    def _load_samples(self):
        raise NotImplementedError
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        image = self._load_image(sample['image_path'])
        
        # Load mask (if available)
        mask = None
        if 'mask_path' in sample and sample['mask_path'] is not None:
            mask = self._load_mask(sample['mask_path'])
        
        # Prepare output
        output = {
            'image': image,
            'label': sample['label'],
            'image_path': str(sample['image_path'])
        }
        
        if mask is not None:
            output['mask'] = mask
        
        if 'defect_type' in sample:
            output['defect_type'] = sample['defect_type']
        
        return output


# ============================================================================
# MVTec AD Dataset
# ============================================================================

class MVTecDataset(AnomalyDataset):
    CATEGORIES = [
        'bottle', 'cable', 'capsule', 'carpet', 'grid',
        'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
        'tile', 'toothbrush', 'transistor', 'wood', 'zipper'
    ]
    
    def __init__(self, root_dir, category, split='train', transform=None, mask_transform=None):
        if category not in self.CATEGORIES:
            raise ValueError(
                f"Invalid category: {category}. "
                f"Available: {self.CATEGORIES}"
            )
        super().__init__(root_dir, category, split, transform, mask_transform)
    
    def _load_samples(self):
        category_dir = self.root_dir / self.category / self.split
        
        if self.split == 'train':
            # Train: only good images
            good_dir = category_dir / 'good'
            if good_dir.exists():
                for img_path in sorted(good_dir.glob('*.png')):
                    self.samples.append({
                        'image_path': img_path,
                        'label': 0,  # Normal
                        'defect_type': 'good'
                    })
        
        else:  # test
            # Good images
            good_dir = category_dir / 'good'
            if good_dir.exists():
                for img_path in sorted(good_dir.glob('*.png')):
                    self.samples.append({
                        'image_path': img_path,
                        'label': 0,  # Normal
                        'defect_type': 'good',
                        'mask_path': None
                    })
            
            # Defect images
            ground_truth_dir = self.root_dir / self.category / 'ground_truth'
            for defect_dir in sorted(category_dir.iterdir()):
                if defect_dir.is_dir() and defect_dir.name != 'good':
                    defect_type = defect_dir.name
                    for img_path in sorted(defect_dir.glob('*.png')):
                        # Find corresponding mask
                        mask_path = None
                        if ground_truth_dir.exists():
                            mask_name = img_path.stem + '_mask.png'
                            potential_mask = ground_truth_dir / defect_type / mask_name
                            if potential_mask.exists():
                                mask_path = potential_mask
                        
                        self.samples.append({
                            'image_path': img_path,
                            'label': 1,  # Anomaly
                            'defect_type': defect_type,
                            'mask_path': mask_path
                        })


# ============================================================================
# VisA Dataset
# ============================================================================

class VisADataset(AnomalyDataset):
    CATEGORIES = [
        'candle', 'capsules', 'cashew', 'chewinggum', 'fryum',
        'macaroni1', 'macaroni2', 'pcb1', 'pcb2', 'pcb3',
        'pcb4', 'pipe_fryum'
    ]
    
    def __init__(self, root_dir, category, split='train', transform=None, mask_transform=None):
        if category not in self.CATEGORIES:
            raise ValueError(
                f"Invalid category: {category}. "
                f"Available: {self.CATEGORIES}"
            )
        super().__init__(root_dir, category, split, transform, mask_transform)
    
    def _load_samples(self):
        split_name = '1cls' if self.split == 'train' else 'test'
        category_dir = self.root_dir / split_name / self.category
        
        if self.split == 'train':
            # Train: only normal images
            for img_path in sorted(category_dir.glob('*.JPG')):
                self.samples.append({
                    'image_path': img_path,
                    'label': 0,  # Normal
                    'defect_type': 'good'
                })
        
        else:  # test
            # Load all test images
            ground_truth_dir = self.root_dir / 'ground_truth' / self.category
            
            for img_path in sorted(category_dir.glob('*.JPG')):
                # Check if mask exists
                mask_name = img_path.stem + '.png'
                mask_path = ground_truth_dir / mask_name if ground_truth_dir.exists() else None
                
                if mask_path and mask_path.exists():
                    label = 1  # Anomaly
                    defect_type = 'defect'
                else:
                    label = 0  # Normal
                    defect_type = 'good'
                    mask_path = None
                
                self.samples.append({
                    'image_path': img_path,
                    'label': label,
                    'defect_type': defect_type,
                    'mask_path': mask_path
                })


# ============================================================================
# BTAD Dataset
# ============================================================================

class BTADDataset(AnomalyDataset):
    CATEGORIES = ['01', '02', '03']
    
    def __init__(self, root_dir, category, split='train', transform=None, mask_transform=None):
        if category not in self.CATEGORIES:
            raise ValueError(
                f"Invalid category: {category}. "
                f"Available: {self.CATEGORIES}"
            )
        super().__init__(root_dir, category, split, transform, mask_transform)
    
    def _load_samples(self):
        category_dir = self.root_dir / self.category
        
        if self.split == 'train':
            # Train: ok directory
            train_dir = category_dir / 'train' / 'ok'
            if train_dir.exists():
                for img_path in sorted(train_dir.glob('*.png')) + sorted(train_dir.glob('*.bmp')):
                    self.samples.append({
                        'image_path': img_path,
                        'label': 0,  # Normal
                        'defect_type': 'good'
                    })
        
        else:  # test
            test_dir = category_dir / 'test'
            ground_truth_dir = category_dir / 'ground_truth'
            
            # Good images
            ok_dir = test_dir / 'ok'
            if ok_dir.exists():
                for img_path in sorted(ok_dir.glob('*.png')) + sorted(ok_dir.glob('*.bmp')):
                    self.samples.append({
                        'image_path': img_path,
                        'label': 0,  # Normal
                        'defect_type': 'good',
                        'mask_path': None
                    })
            
            # Defect images
            for defect_dir in sorted(test_dir.iterdir()):
                if defect_dir.is_dir() and defect_dir.name != 'ok':
                    defect_type = defect_dir.name
                    for img_path in sorted(defect_dir.glob('*.png')) + sorted(defect_dir.glob('*.bmp')):
                        # Find corresponding mask
                        mask_path = None
                        if ground_truth_dir.exists():
                            mask_name = img_path.name
                            potential_mask = ground_truth_dir / defect_type / mask_name
                            if potential_mask.exists():
                                mask_path = potential_mask
                        
                        self.samples.append({
                            'image_path': img_path,
                            'label': 1,  # Anomaly
                            'defect_type': defect_type,
                            'mask_path': mask_path
                        })


# ============================================================================
# Custom Dataset (OLED)
# ============================================================================

class CustomDataset(AnomalyDataset):
    pass