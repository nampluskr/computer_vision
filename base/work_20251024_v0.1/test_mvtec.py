"""" filename: test_mvtec.py """

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision
import numpy as np
import os
import glob
from PIL import Image
from time import time
from trainer import BaseTrainer, EarlyStopper, set_seed, set_logger


class MVTecDataset(Dataset):
    def __init__(self, root_dir, category="bottle", split="train", transform=None, mask_transform=None):
        self.root_dir = root_dir
        self.category = category
        self.split = split
        self.img_size = 256

        if transform is None:
            self.transform = T.Compose([
                T.Resize((self.img_size, self.img_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform

        if mask_transform is None:
            self.mask_transform = T.Compose([
                T.Resize((self.img_size, self.img_size)),
                T.ToTensor()
            ])
        else:
            self.mask_transform = mask_transform

        self.image_paths = []
        self.mask_paths = []
        self.labels = []

        self._load_data()

    def _load_data(self):
        category_dir = os.path.join(self.root_dir, self.category, self.split)

        if self.split == "train":
            normal_dir = os.path.join(category_dir, "good")
            if os.path.exists(normal_dir):
                image_files = sorted(glob.glob(os.path.join(normal_dir, "*.png")))
                for img_path in image_files:
                    self.image_paths.append(img_path)
                    self.mask_paths.append(None)
                    self.labels.append(0)
        else:
            if not os.path.exists(category_dir):
                return

            defect_dirs = sorted([d for d in os.listdir(category_dir)
                                if os.path.isdir(os.path.join(category_dir, d))])

            for defect_name in defect_dirs:
                defect_dir = os.path.join(category_dir, defect_name)
                is_good = defect_name == "good"
                label = 0 if is_good else 1

                image_files = sorted(glob.glob(os.path.join(defect_dir, "*.png")))
                for img_path in image_files:
                    self.image_paths.append(img_path)

                    if is_good:
                        self.mask_paths.append(None)
                    else:
                        img_name = os.path.splitext(os.path.basename(img_path))[0]
                        mask_dir = os.path.join(self.root_dir, self.category, "ground_truth", defect_name)
                        mask_path = os.path.join(mask_dir, img_name + "_mask.png")
                        self.mask_paths.append(mask_path if os.path.exists(mask_path) else None)

                    self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        image = self.transform(image)

        label = torch.tensor(self.labels[idx]).long()

        if self.mask_paths[idx] is not None:
            mask = Image.open(self.mask_paths[idx]).convert("L")
            mask = self.mask_transform(mask)
        else:
            mask = torch.zeros((1, self.img_size, self.img_size))

        return dict(image=image, label=label, mask=mask)


class SimpleAutoencoder(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(256 * 16 * 16, latent_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256 * 16 * 16),
            nn.ReLU(inplace=True),
            nn.Unflatten(1, (256, 16, 16)),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1)
        )

    def forward(self, images):
        latent = self.encoder(images)
        reconstructed = self.decoder(latent)
        
        if self.training:
            return reconstructed, latent
        
        anomaly_map = torch.mean((images - reconstructed)**2, dim=1, keepdim=True)
        pred_score = torch.amax(anomaly_map, dim=(-2, -1))
        return dict(pred_score=pred_score, anomaly_map=anomaly_map)


class AnomalyTrainer(BaseTrainer):
    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)
        self.ssim_metric = StructuralSimilarityIndexMeasure().to(self.device)
        self.auroc_metric = BinaryAUROC().to(self.device)
        self.aupr_metric = BinaryAveragePrecision().to(self.device)

    def on_validation_epoch_start(self):
        super().on_validation_epoch_start()
        self.auroc_metric.reset()
        self.aupr_metric.reset()

    def training_step(self, batch, batch_idx):
        images = batch['image'].to(self.device)

        reconstructed, latent = self.model(images)
        loss = self.loss_fn(reconstructed, images)
        ssim_value = self.ssim_metric(reconstructed, images)
        return dict(loss=loss, ssim=ssim_value)

    def validation_step(self, batch, batch_idx):
        images = batch['image'].to(self.device)
        labels = batch['label'].to(self.device)

        outputs = self.model(images)
        anomaly_scores = outputs['pred_score'].squeeze()

        self.auroc_metric.update(anomaly_scores, labels)
        self.aupr_metric.update(anomaly_scores, labels)
        return dict(auroc=torch.tensor(0.0), aupr=torch.tensor(0.0))

    def on_validation_epoch_end(self, outputs):
        auroc_value = self.auroc_metric.compute()
        aupr_value = self.aupr_metric.compute()
        
        outputs['auroc'] = auroc_value.item()
        outputs['aupr'] = aupr_value.item()
        
        super().on_validation_epoch_end(outputs)
        
        self.auroc_metric.reset()
        self.aupr_metric.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        return dict(optimizer=optimizer, scheduler=scheduler)

    def configure_early_stoppers(self):
        return dict(
            train=EarlyStopper(patience=5, mode='min', min_delta=0.001, monitor='loss'),
            valid=EarlyStopper(patience=10, mode='max', min_delta=0.001, target_value=0.95, monitor='auroc')
        )


def main():
    root_dir = "/mnt/d/datasets/mvtec"
    category = "bottle"
    batch_size = 16
    num_epochs = 10
    output_dir = "./outputs_mvtec"
    run_name = f"mvtec_{category}_ae"
    seed = 42

    set_seed(seed)
    logger = set_logger(output_dir, run_name)

    logger.info("Loading MVTec dataset...")
    train_dataset = MVTecDataset(root_dir, category=category, split="train")
    valid_dataset = MVTecDataset(root_dir, category=category, split="test")
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Valid samples: {len(valid_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)

    valid_loader = DataLoader(valid_dataset, batch_size=batch_size,
        shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True)

    model = SimpleAutoencoder(latent_dim=256)
    loss_fn = nn.MSELoss()
    trainer = AnomalyTrainer(model, loss_fn=loss_fn, logger=logger)
    trainer.fit(train_loader, num_epochs, valid_loader=valid_loader, 
                output_dir=output_dir, run_name=run_name)


if __name__ == "__main__":
    main()