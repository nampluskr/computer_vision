"""" filename: test_mvtec.py """

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision
import numpy as np
import os
import glob
from PIL import Image
from time import time
from trainer import BaseTrainer, EarlyStopper


#############################################################
# MVTec AD Dataset
#############################################################

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


#############################################################
# Simple Autoencoder Model
#############################################################

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

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed


#############################################################
# Anomaly Trainer (FIX #1 적용)
#############################################################

class AnomalyTrainer(BaseTrainer):
    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)
        self.ssim_metric = StructuralSimilarityIndexMeasure().to(self.device)
        self.auroc_metric = BinaryAUROC().to(self.device)
        self.aupr_metric = BinaryAveragePrecision().to(self.device)

    def on_validation_epoch_start(self):
        """Validation epoch 시작 시 metric 초기화"""
        super().on_validation_epoch_start()
        self.auroc_metric.reset()
        self.aupr_metric.reset()

    def training_step(self, batch, batch_idx):
        images = batch['image'].to(self.device)

        reconstructed = self.model(images)
        loss = self.loss_fn(reconstructed, images)
        ssim_value = self.ssim_metric(reconstructed, images)
        return dict(loss=loss, ssim=ssim_value)

    def validation_step(self, batch, batch_idx):
        """
        FIX #1: 각 배치마다 anomaly score를 누적만 하고, 
        최종 metric 계산은 on_validation_epoch_end에서 수행
        """
        images = batch['image'].to(self.device)
        labels = batch['label'].to(self.device)

        reconstructed = self.model(images)
        anomaly_scores = torch.mean((reconstructed - images) ** 2, dim=(1, 2, 3))

        # Metric에 누적만 하고 compute는 하지 않음
        self.auroc_metric.update(anomaly_scores, labels)
        self.aupr_metric.update(anomaly_scores, labels)

        # Progress bar 표시용으로 빈 dict 반환 (또는 다른 metric 반환 가능)
        return dict()

    def on_validation_epoch_end(self, outputs):
        """
        FIX #1: Validation epoch 종료 시 전체 데이터에 대한 최종 metric 계산
        """
        # 전체 epoch에 대한 AUROC, AUPR 계산
        auroc_value = self.auroc_metric.compute()
        aupr_value = self.aupr_metric.compute()
        
        # History 저장을 위해 수동으로 추가
        if 'auroc' not in self.history['valid']:
            self.history['valid']['auroc'] = []
        if 'aupr' not in self.history['valid']:
            self.history['valid']['aupr'] = []
        
        self.history['valid']['auroc'].append(auroc_value.item())
        self.history['valid']['aupr'].append(aupr_value.item())
        
        # 로그 출력
        self.logger.info(f"auroc:{auroc_value:.3f}, aupr:{aupr_value:.3f}")
        
        # 다음 epoch을 위해 metric 초기화
        self.auroc_metric.reset()
        self.aupr_metric.reset()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        return dict(optimizer=optimizer, scheduler=scheduler)

    def configure_early_stoppers(self):
        return dict(
            train=EarlyStopper(patience=5, mode='min', min_delta=0.001),
            valid=EarlyStopper(patience=10, mode='max', min_delta=0.001, target_value=0.95)
        )


#############################################################
# Main
#############################################################

def main():
    root_dir = "/mnt/d/datasets/mvtec"
    category = "bottle"
    batch_size = 32
    num_epochs = 20
    output_dir = "./logs_mvtec"

    train_dataset = MVTecDataset(root_dir, category=category, split="train")
    valid_dataset = MVTecDataset(root_dir, category=category, split="test")

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)

    valid_loader = DataLoader(valid_dataset, batch_size=batch_size,
        shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True)

    model = SimpleAutoencoder(latent_dim=256)
    loss_fn = nn.MSELoss()
    trainer = AnomalyTrainer(model, loss_fn=loss_fn)

    trainer.fit(train_loader, num_epochs, valid_loader=valid_loader, output_dir=output_dir)
    
    # 최종 모델 저장 (best_model.pth는 자동으로 저장됨)
    trainer.save_model(os.path.join(output_dir, f"{category}_final.pth"))


if __name__ == "__main__":
    main()