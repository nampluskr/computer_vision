import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T



DATASET_DIR = ""    # "/mnt/d/datasets"

def get_dataset_dir():
    return DATASET_DIR

def set_dataset_dir(dataset_dir):
    global DATASET_DIR
    DATASET_DIR = dataset_dir
    print(f" > Dataset  directory set to: {DATASET_DIR}")


#####################################################################
# Base Dataset
#####################################################################

class BaseDataset(Dataset):
    def __init__(self, dataset_dir, dataset_type, category, split="train",
                 transform=None, mask_transform=None, **kwargs):
        super().__init__()
        self.transform = transform
        self.mask_transform = mask_transform
        self.image_paths = []
        self.mask_paths = []
        self.labels = []
        self.defect_types = []
        self.has_mask = True    # True for MVTec/VisA/BTAD
        self.categories = []

        self.load_data(dataset_dir, dataset_type, category, split, **kwargs)

        dtype_str = "-".join(dataset_type) if isinstance(dataset_type, list) else dataset_type
        if isinstance(category, list):
            cat_str = "-".join(category)
        elif category == "all":
            cat_str = "all"
        else:
            cat_str = category

        print(f" > {dtype_str}/{cat_str} | {split} set: {len(self)} images, "
              f"Normal: {self.labels.count(0)}, Anomaly: {self.labels.count(1)}")

    def load_data(self, dataset_dir, category, split, **kwargs):
        raise NotImplementedError

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        height, width = image.shape[-2:]

        if label == 0 or self.mask_paths[idx] is None:
            mask = np.zeros((height, width), dtype=np.uint8)
        else:
            mask_path = self.mask_paths[idx]
            mask = Image.open(mask_path).convert('L')
            if self.mask_transform:
                mask = self.mask_transform(mask)
            mask = (np.array(mask) > 0).astype(np.uint8)
        mask = torch.tensor(mask).long()

        label = torch.tensor(label).long()
        name = os.path.basename(image_path)
        defect_type = self.defect_types[idx]
        category = self.categories[idx]
        return dict(image=image, label=label, mask=mask, name=name, category=category,
                    defect_type=defect_type, has_mask=self.has_mask)


#####################################################################
# Custom Dataset
#####################################################################

def get_data_info(image_path):
    basename = os.path.basename(image_path)
    filename_parts = os.path.splitext(basename)[0].split()

    if len(filename_parts) != 3:
        raise ValueError(f"Invalid filename format: {image_path}")

    info = {}
    info["filename"] = basename
    info["category"] = filename_parts[0]
    info["freq"] = int(filename_parts[1])
    info["dimming"] = int(filename_parts[2].split('_')[0])
    info["image_path"] = image_path
    info["dataset_type"] = None
    info["defect_type"] = None
    info["label"] = None

    path_parts = os.path.normpath(image_path).split(os.sep)
    if "data_rgb" in path_parts:
        idx = path_parts.index("data_rgb")
        info["dataset_type"] = path_parts[idx - 1] if idx > 0 else None

        if idx + 1 < len(path_parts):
            defect_type = path_parts[idx + 1]
            info["defect_type"] = defect_type
            info["label"] = 0 if defect_type == "normal" else 1
    return info


def create_csv(csv_path):
    data_dir = os.path.dirname(csv_path)
    records = []
    for subfolder in os.listdir(data_dir):
        subfolder_path = os.path.join(data_dir, subfolder)

        if not os.path.isdir(subfolder_path):
            continue

        for filename in sorted(os.listdir(subfolder_path)):
            if not filename.endswith(".png"):
                continue

            image_path = os.path.join(subfolder_path, filename)
            try:
                info = get_data_info(image_path)
                records.append(info)
            except Exception as e:
                print(f"Warning: Error parsing {image_path}: {e}")
                continue

    if not records:
        raise ValueError(f"No valid images found in {data_dir}")

    df = pd.DataFrame(records)
    df.to_csv(csv_path, index=False)

    print(f" > Total images: {len(df)}")
    print(f" > Categories: {df['category'].unique()}")
    print(f" > Labels: Normal={len(df[df['label']==0])}, Anomaly={len(df[df['label']==1])}")


class CustomDataset(BaseDataset):
    def load_data(self, dataset_dir, dataset_type, category, split="train", test_ratio=0.2):
        self.has_mask = False
        dataset_types = [dataset_type] if isinstance(dataset_type, str) else dataset_type

        all_dfs = []
        for dtype in dataset_types:
            data_dir = os.path.join(dataset_dir, dtype, "data_rgb")
            csv_path = os.path.join(data_dir, "data_info.csv")

            if not os.path.exists(csv_path):
                print(f" > CSV file not found: {csv_path}")
                create_csv(csv_path)
                print(f" > CSV file created: {csv_path}")

            df = pd.read_csv(csv_path)
            all_dfs.append(df)

        df = pd.concat(all_dfs, axis=0, ignore_index=True)

        if isinstance(category, str):
            categories = None if category == "all" else [category]
        else:
            categories = list(category)

        if categories is not None:
            df = df[df["category"].isin(categories)].reset_index(drop=True)
            if len(df) == 0:
                all_patterns = pd.concat(all_dfs, axis=0)["category"].unique()
                raise ValueError(
                    f"No images found for patterns {categories}. "
                    f"Available patterns: {list(all_patterns)}"
                )
        normal_df = df[df["label"] == 0].reset_index(drop=True)
        anomaly_df = df[df["label"] == 1].reset_index(drop=True)

        if split == "train":
            if len(normal_df) == 0:
                raise ValueError(f"No normal samples found for category '{category}'")
            train_normal, _ = train_test_split(normal_df,
                test_size=test_ratio, random_state=42, shuffle=True)
            subset = train_normal
        elif split == "test":
            if len(normal_df) == 0:
                raise ValueError(f"No normal samples found for category '{category}'")
            _, test_normal = train_test_split(normal_df,
                test_size=test_ratio, random_state=42, shuffle=True)
            subset = pd.concat([test_normal, anomaly_df], axis=0).reset_index(drop=True)
        else:
            raise ValueError(f"Invalid split: {split}. Expected 'train' or 'test'")

        for _, row in subset.iterrows():
            image_path = row["image_path"]
            if not os.path.exists(image_path):
                print(f"Warning: Image not found: {image_path}")
                continue

            self.image_paths.append(image_path)
            self.labels.append(int(row["label"]))
            self.defect_types.append(row["defect_type"])
            self.mask_paths.append(None)
            self.categories.append(row["category"])


#####################################################################
# MVTec Dataset
#####################################################################

class MVTecDataset(BaseDataset):
    def load_data(self, dataset_dir, dataset_type, category, split, **kwargs):
        self.has_mask = True
        category_dir = os.path.join(dataset_dir, dataset_type, category)

        if split == "train":
            normal_image_dir = os.path.join(category_dir, "train", "good")
            for image_name in sorted(os.listdir(normal_image_dir)):
                normal_image_path = os.path.join(normal_image_dir, image_name)
                self.image_paths.append(normal_image_path)
                self.mask_paths.append(None)
                self.labels.append(0)
                self.defect_types.append("good")
                self.categories.append(category)

        elif split == "test":
            normal_image_dir = os.path.join(category_dir, "test", "good")
            for image_name in sorted(os.listdir(normal_image_dir)):
                normal_image_path = os.path.join(normal_image_dir, image_name)
                self.image_paths.append(normal_image_path)
                self.mask_paths.append(None)
                self.labels.append(0)
                self.defect_types.append("good")
                self.categories.append(category)

            test_dir = os.path.join(category_dir, "test")
            for defect_type in sorted(os.listdir(test_dir)):
                if defect_type == "good": continue

                anomaly_image_dir = os.path.join(test_dir, defect_type)
                anomaly_mask_dir = os.path.join(category_dir, "ground_truth", defect_type)
                for image_name in sorted(os.listdir(anomaly_image_dir)):
                    anomaly_image_path = os.path.join(anomaly_image_dir, image_name)
                    image_stem = os.path.splitext(image_name)[0]
                    anomaly_mask_path = os.path.join(anomaly_mask_dir, f"{image_stem}_mask.png")
                    self.image_paths.append(anomaly_image_path)
                    self.mask_paths.append(anomaly_mask_path)
                    self.labels.append(1)
                    self.defect_types.append(defect_type)
                    self.categories.append(category)


#####################################################################
# VisA Dataset
#####################################################################

class VisADataset(BaseDataset):
    def load_data(self, dataset_dir, dataset_type, category, split="train", test_ratio=0.2):
        self.has_mask = True
        category_dir = os.path.join(dataset_dir, dataset_type, category)
        csv_path = os.path.join(category_dir, "image_anno.csv")
        df = pd.read_csv(csv_path)

        normal_df = df[df["label"] == "normal"].reset_index(drop=True)
        anomaly_df = df[df["label"] != "normal"].reset_index(drop=True)

        if split == "train":
            normal_train, _ = train_test_split(normal_df,
                test_size=test_ratio, random_state=42, shuffle=True)
            subset = normal_train
        elif split == "test":
            _, normal_test = train_test_split(normal_df,
                test_size=test_ratio, random_state=42, shuffle=True)
            subset = pd.concat([normal_test, anomaly_df], axis=0).reset_index(drop=True)

        for _, row in subset.iterrows():
            image_path = os.path.join(dataset_dir, dataset_type, row["image"])
            defect_type = row["label"]

            if defect_type == "normal":
                self.image_paths.append(image_path)
                self.mask_paths.append(None)
                self.labels.append(0)
                self.defect_types.append("good")
                self.categories.append(category)
            else:
                self.image_paths.append(image_path)
                mask_path = os.path.join(dataset_dir, dataset_type, row["mask"])
                self.mask_paths.append(mask_path)
                self.labels.append(1)
                self.defect_types.append(defect_type)
                self.categories.append(category)


#####################################################################
# BTAD Dataset
#####################################################################

class BTADDataset(BaseDataset):
    def load_data(self, dataset_dir, dataset_type, category, split, **kwargs):
        self.has_mask = True
        category_dir = os.path.join(dataset_dir, dataset_type, category)

        if split == "train":
            normal_image_dir = os.path.join(category_dir, "train", "ok")
            for image_name in sorted(os.listdir(normal_image_dir)):
                normal_image_path = os.path.join(normal_image_dir, image_name)
                self.image_paths.append(normal_image_path)
                self.mask_paths.append(None)
                self.labels.append(0)
                self.defect_types.append("good")
                self.categories.append(category)

        elif split == "test":
            normal_image_dir = os.path.join(category_dir, "test", "ok")
            for image_name in sorted(os.listdir(normal_image_dir)):
                normal_image_path = os.path.join(normal_image_dir, image_name)
                self.image_paths.append(normal_image_path)
                self.mask_paths.append(None)
                self.labels.append(0)
                self.defect_types.append("good")
                self.categories.append(category)

            anomaly_image_dir = os.path.join(category_dir, "test", "ko")
            anomaly_mask_dir = os.path.join(category_dir, "ground_truth", "ko")
            for image_name in sorted(os.listdir(anomaly_image_dir)):
                anomaly_image_path = os.path.join(anomaly_image_dir, image_name)
                image_stem = os.path.splitext(image_name)[0]
                anomaly_mask_path = None

                for ext in ['.png', '.bmp']:
                    candidate = os.path.join(anomaly_mask_dir, image_stem + ext)
                    if os.path.exists(candidate):
                        anomaly_mask_path = candidate
                        break

                self.image_paths.append(anomaly_image_path)
                self.mask_paths.append(anomaly_mask_path)
                self.labels.append(1)
                self.defect_types.append("anomaly")
                self.categories.append(category)


#####################################################################
# Data Loader for MVTec / VisA / BTAD / Custom Datasets
#####################################################################

def get_dataloaders(dataset_dir, dataset_type, category, img_size, batch_size, normalize=True,
                    test_ratio=0.2, num_workers=8, pin_memory=True, persistent_workers=True):
    train_transforms = [
        T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BILINEAR),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.ColorJitter(
            brightness=(0.8, 1.2),  # Brightness factor between 0.8 and 1.2
            contrast=(0.7, 1.3),    # Contrast factor between 0.7 and 1.3
            saturation=(0.7, 1.3),  # Saturation factor between 0.7 and 1.3
            hue=(-0.1, 0.1)         # Hue shift between -0.1 and 0.1
        ),
        T.ToTensor(),
    ]
    test_transforms = [
        T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
    ]
    if normalize:
        normalize_transform = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        train_transforms.append(normalize_transform)
        test_transforms.append(normalize_transform)

    train_transform = T.Compose(train_transforms)
    test_transform = T.Compose(test_transforms)

    mask_transform = T.Compose([
        T.Resize((img_size, img_size), interpolation=T.InterpolationMode.NEAREST),
    ])

    datasets = {
        "mvtec": MVTecDataset,
        "visa": VisADataset,
        "btad": BTADDataset
    }
    if isinstance(dataset_type, list) or isinstance(category, list):
        DatasetClass = CustomDataset
    else:
        DatasetClass = datasets.get(dataset_type.lower(), CustomDataset)

    train_set = DatasetClass(dataset_dir, dataset_type, category, split="train",
        transform=train_transform, mask_transform=mask_transform, test_ratio=test_ratio)
    test_set = DatasetClass(dataset_dir, dataset_type, category, split="test",
        transform=test_transform, mask_transform=mask_transform, test_ratio=test_ratio)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers)
    return train_loader, test_loader