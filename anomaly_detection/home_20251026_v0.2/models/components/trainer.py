""" filename: models/components/trainer.py """

from abc import ABC, abstractmethod
from tqdm import tqdm
from time import time
from copy import deepcopy
import os
import logging
import warnings

import torch
import torch.nn as nn
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision


# ============================================================================
# Early Stopper
# ============================================================================

class EarlyStopper:
    def __init__(self, patience=10, min_delta=1e-3, mode='max', target_value=None, monitor='loss'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.target_value = target_value
        self.monitor = monitor
        self.reset()

    def reset(self):
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.target_reached = False

    def __call__(self, score):
        if self.target_value is not None:
            if self.mode == 'max' and score >= self.target_value:
                self.target_reached = True
                self.early_stop = True
                return True
            if self.mode == 'min' and score <= self.target_value:
                self.target_reached = True
                self.early_stop = True
                return True

        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True
            return True

        return False


# ============================================================================
# Base Trainer
# ============================================================================

class BaseTrainer(ABC):
    def __init__(self, model, loss_fn=None, device=None, logger=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if model is None:
            raise ValueError("Model must be provided")
        self.model = model.to(self.device)
        self.loss_fn = loss_fn.to(self.device) if isinstance(loss_fn, nn.Module) else loss_fn

        self.global_step = 0
        self.global_epoch = 0
        self.training = True
        self.best_model_state = None
        self.history = {"train": {}, "valid": {}}

        if logger is None:
            console_logger = logging.getLogger(f"Trainer_{id(self)}")
            console_logger.setLevel(logging.INFO)
            if not console_logger.handlers:
                console_handler = logging.StreamHandler()
                console_handler.setFormatter(logging.Formatter('%(message)s'))
                console_logger.addHandler(console_handler)
            self.logger = console_logger
        else:
            self.logger = logger

    @abstractmethod
    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    @abstractmethod
    def validation_step(self, batch, batch_idx):
        raise NotImplementedError

    def configure_optimizers(self):
        return None

    def configure_early_stoppers(self):
        return None

    def on_train_epoch_start(self):
        self.model.train()
        self.training = True

        self.global_epoch += 1
        self.epoch_start_time = time()
        self.epoch_info = f"[{self.epoch:3d}/{self.num_epochs}]"

    def on_train_epoch_end(self, outputs):
        self._update_history(outputs)
        self.train_info = ", ".join([f"{k}:{v:.3f}" for k, v in outputs.items()])
        if not self.has_valid_loader:
            time_info = self._format_time(time() - self.epoch_start_time)
            self.logger.info(f"{self.epoch_info} {self.train_info} ({time_info})")

    def on_validation_epoch_start(self):
        self.model.eval()
        self.training = False

    def on_validation_epoch_end(self, outputs):
        self._update_history(outputs)
        valid_info = ", ".join([f"{k}:{v:.3f}" for k, v in outputs.items()])
        time_info = self._format_time(time() - self.epoch_start_time)
        self.logger.info(f"{self.epoch_info} {self.train_info} | (val) {valid_info} ({time_info})")

    def on_train_batch_start(self, batch, batch_idx): pass
    def on_train_batch_end(self, outputs, batch, batch_idx): pass

    def on_validation_batch_start(self, batch, batch_idx): pass
    def on_validation_batch_end(self, outputs, batch, batch_idx): pass

    def on_fit_start(self):
        if self.output_dir is not None:
            os.makedirs(self.output_dir, exist_ok=True)
        self.fit_start_time = time()
        self._setup_optimizers()
        self._setup_early_stoppers()

        self.logger.info(f"Starting training on device: {self.device}")
        self.logger.info(f"Model: {self.model.__class__.__name__}")
        self.logger.info("-" * 70)

    def on_fit_end(self):
        self.total_time = time() - self.fit_start_time
        self.logger.info("-" * 70)
        self.logger.info(f"Total training time: {self._format_time(self.total_time)}")

        if self.best_model_state is not None and self.output_dir is not None:
            best_model_path = os.path.join(self.output_dir, f"{self.run_name}_epoch-{self.global_epoch}.pth")
            torch.save(self.best_model_state, best_model_path)
            self.logger.info(f"Best model saved to: {best_model_path}")

        self.logger.info("Training completed!")

    def _update_history(self, outputs):
        history_key = 'train' if self.training else 'valid'
        for key, value in outputs.items():
            self.history[history_key].setdefault(key, [])
            self.history[history_key][key].append(value)

    def _setup_optimizers(self):
        optimizer_config = self.configure_optimizers()
        if optimizer_config is None:
            self.optimizer = None
            self.scheduler = None
        elif isinstance(optimizer_config, dict):
            self.optimizer = optimizer_config['optimizer']
            self.scheduler = optimizer_config.get('scheduler', None)
        else:
            self.optimizer = optimizer_config
            self.scheduler = None

    def _setup_early_stoppers(self):
        early_stopper_config = self.configure_early_stoppers()
        if early_stopper_config is None:
            self.train_early_stopper = None
            self.valid_early_stopper = None
        elif isinstance(early_stopper_config, dict):
            self.train_early_stopper = early_stopper_config.get('train', None)
            self.valid_early_stopper = early_stopper_config.get('valid', None)
        else:
            self.train_early_stopper = None
            self.valid_early_stopper = early_stopper_config

    def _format_time(self, seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        if hours > 0:
            return f"{hours}h {minutes}m {secs}s"
        elif minutes > 0:
            return f"{minutes}m {secs}s"
        else:
            return f"{secs}s"

    def _update_scheduler(self):
        if self.scheduler is None:
            return

        old_lrs = [group['lr'] for group in self.optimizer.param_groups]
        self.scheduler.step()
        new_lrs = [group['lr'] for group in self.optimizer.param_groups]

        lr_threshold = 1e-8
        if len(new_lrs) == 1:
            if abs(old_lrs[0] - new_lrs[0]) > lr_threshold:
                self.logger.info(f"Learning rate updated: {old_lrs[0]:.3e} -> {new_lrs[0]:.3e}")
        else:
            for idx, (old_lr, new_lr) in enumerate(zip(old_lrs, new_lrs)):
                if abs(old_lr - new_lr) > lr_threshold:
                    self.logger.info(f"Learning rate [group {idx}] updated: {old_lr:.3e} -> {new_lr:.3e}")

    def _check_train_stopping(self, train_outputs):
        if self.train_early_stopper is None:
            return False

        monitor_key = self.train_early_stopper.monitor
        if monitor_key not in train_outputs:
            self.logger.error(f"Monitor key '{monitor_key}' not found in train outputs."
                              f" Available: {list(train_outputs.keys())}")
            return False

        score = train_outputs[monitor_key]
        best_score = self.train_early_stopper.best_score
        if self.train_early_stopper(score):
            if self.train_early_stopper.target_reached:
                self.logger.info(f"Target train {monitor_key} reached: {score:.3f}")
            else:
                self.logger.info(f"Early stopping - Best train {monitor_key}: {best_score:.3f}")
            return True
        return False

    def _check_valid_stopping(self, valid_outputs):
        if self.valid_early_stopper is None:
            return False

        monitor_key = self.valid_early_stopper.monitor
        if monitor_key not in valid_outputs:
            self.logger.error(f"Monitor key '{monitor_key}' not found in valid outputs."
                              f" Available: {list(valid_outputs.keys())}")
            return False

        score = valid_outputs[monitor_key]
        best_score = self.valid_early_stopper.best_score
        is_best = False
        if best_score is None:
            is_best = True
        elif self.valid_early_stopper.mode == 'max':
            is_best = score > best_score
        else:
            is_best = score < best_score

        if is_best:
            self.best_model_state = deepcopy(self.model.state_dict())

        if self.valid_early_stopper(score):
            if self.valid_early_stopper.target_reached:
                self.logger.info(f"Target valid {monitor_key} reached: {score:.3f}")
            else:
                self.logger.info(f"Early stopping - Best valid {monitor_key}: {best_score:.3f}")
            return True
        return False

    @torch.enable_grad()
    def _train_epoch(self, train_loader):
        accumulated_outputs = {}
        total_images = 0

        with tqdm(train_loader, leave=False, ascii=True) as progress_bar:
            progress_bar.set_description(f"{self.epoch_info} Training")
            for batch_idx, batch in enumerate(progress_bar):
                batch_size = batch["image"].shape[0]
                total_images += batch_size

                self.on_train_batch_start(batch, batch_idx)

                if self.optimizer is not None:
                    self.optimizer.zero_grad()
                    outputs = self.training_step(batch, batch_idx)
                    loss = outputs.get('loss')

                    if loss is not None:
                        loss.backward()
                        self.optimizer.step()
                    self.global_step += 1
                else:
                    outputs = self.training_step(batch, batch_idx)

                for name, value in outputs.items():
                    if isinstance(value, torch.Tensor):
                        value = value.item()
                    accumulated_outputs.setdefault(name, 0.0)
                    accumulated_outputs[name] += value * batch_size

                progress_bar.set_postfix({
                    name: f"{value / total_images:.3f}"
                    for name, value in accumulated_outputs.items()
                })
                self.on_train_batch_end(outputs, batch, batch_idx)

        return {name: value / total_images for name, value in accumulated_outputs.items()}

    @torch.no_grad()
    def _validate_epoch(self, valid_loader):
        accumulated_outputs = {}
        total_images = 0

        with tqdm(valid_loader, leave=False, ascii=True) as progress_bar:
            progress_bar.set_description(f"{self.epoch_info} Validation")
            for batch_idx, batch in enumerate(progress_bar):
                batch_size = batch["image"].shape[0]
                total_images += batch_size

                self.on_validation_batch_start(batch, batch_idx)
                outputs = self.validation_step(batch, batch_idx)

                for name, value in outputs.items():
                    if isinstance(value, torch.Tensor):
                        value = value.item()
                    accumulated_outputs.setdefault(name, 0.0)
                    accumulated_outputs[name] += value * batch_size

                progress_bar.set_postfix({
                    name: f"{total_value / total_images:.3f}"
                    for name, total_value in accumulated_outputs.items()
                })
                self.on_validation_batch_end(outputs, batch, batch_idx)

        return {name: value / total_images for name, value in accumulated_outputs.items()}

    def fit(self, train_loader, num_epochs, valid_loader=None, output_dir=None, run_name=None):
        self.has_valid_loader = valid_loader is not None
        self.num_epochs = num_epochs
        self.output_dir = output_dir
        self.run_name = run_name or "best_model"
        self.on_fit_start()

        for epoch in range(1, num_epochs + 1):
            self.epoch = epoch
            self.on_train_epoch_start()
            train_outputs = self._train_epoch(train_loader)

            self.on_train_epoch_end(train_outputs)
            if self._check_train_stopping(train_outputs):
                break

            if self.has_valid_loader:
                self.on_validation_epoch_start()
                valid_outputs = self._validate_epoch(valid_loader)

                self.on_validation_epoch_end(valid_outputs)
                if self._check_valid_stopping(valid_outputs):
                    break

            self._update_scheduler()

        self.on_fit_end()
        return self.history

    def save_checkpoint(self, filepath):
        output_dir = os.path.dirname(filepath)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "global_step": self.global_step,
            "global_epoch": self.global_epoch,
            "history": self.history,
        }
        if self.optimizer is not None:
            checkpoint["optimizer_state_dict"] = self.optimizer.state_dict()

        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        torch.save(checkpoint, filepath)
        self.logger.info(f"Checkpoint saved: {filepath}")

    def load_checkpoint(self, filepath, strict=True):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
        self.global_step = checkpoint.get("global_step", 0)
        self.global_epoch = checkpoint.get("global_epoch", 0)
        self.history = checkpoint.get("history", {"train": {}, "valid": {}})

        if "optimizer_state_dict" in checkpoint and self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if "scheduler_state_dict" in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.logger.info(f"Checkpoint loaded: {filepath}")
        self.logger.info(f"Resumed from global_epoch: {self.global_epoch}, global_step: {self.global_step}")

    def save_model(self, filepath):
        output_dir = os.path.dirname(filepath)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        torch.save(self.model.state_dict(), filepath)
        self.logger.info(f"Model saved: {filepath}")

    def load_model(self, filepath, strict=True):
        self.model.load_state_dict(torch.load(filepath, map_location=self.device), strict=strict)
        self.logger.info(f"Model loaded: {filepath}")


# ============================================================================
# Anomaly Trainer
# ============================================================================

class AnomalyTrainer(BaseTrainer):
    """
    Base trainer for all anomaly detection models
    Subclasses must implement training_step() for their specific model
    """

    def __init__(self, model, loss_fn=None, device=None, logger=None):
        super().__init__(model, loss_fn=loss_fn, device=device, logger=logger)

        # Initialize metrics for validation
        self.auroc_metric = BinaryAUROC().to(self.device)
        self.aupr_metric = BinaryAveragePrecision().to(self.device)

    def training_step(self, batch, batch_idx):
        """
        Must be implemented by subclass
        Should return dict with at least 'loss' key
        """
        raise NotImplementedError(
            "training_step must be implemented by subclass. "
            "Return dict with 'loss' and other metrics."
        )

    def on_validation_epoch_start(self):
        super().on_validation_epoch_start()
        self.auroc_metric.reset()
        self.aupr_metric.reset()

    def validation_step(self, batch, batch_idx):
        """
        Validation step for anomaly detection
        Model should return dict with 'pred_score' or 'anomaly_score'
        """
        images = batch['image'].to(self.device)
        labels = batch['label'].to(self.device)

        # Forward pass
        outputs = self.model(images)

        # Get anomaly scores
        if isinstance(outputs, dict):
            anomaly_scores = outputs.get('pred_score', outputs.get('anomaly_score', None))
            if anomaly_scores is None:
                raise ValueError(
                    "Model output must contain 'pred_score' or 'anomaly_score' key. "
                    f"Got keys: {list(outputs.keys())}"
                )
        else:
            # Assume output is anomaly score
            if outputs.dim() > 1:
                anomaly_scores = torch.mean(outputs.view(outputs.size(0), -1), dim=1)
            else:
                anomaly_scores = outputs

        # Ensure anomaly_scores has batch dimension [B] not []
        if anomaly_scores.dim() == 0:
            # Scalar case: add batch dimension
            anomaly_scores = anomaly_scores.unsqueeze(0)
        elif anomaly_scores.dim() > 1:
            # [B, 1, ...] -> [B]
            anomaly_scores = anomaly_scores.view(anomaly_scores.size(0), -1).mean(dim=1)

        # Ensure labels has same shape [B]
        if labels.dim() == 0:
            labels = labels.unsqueeze(0)
        elif labels.dim() > 1:
            labels = labels.view(-1)

        # Update metrics (accumulate across batches)
        self.auroc_metric.update(anomaly_scores, labels)
        self.aupr_metric.update(anomaly_scores, labels)

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                auroc = self.auroc_metric.compute()
                aupr = self.aupr_metric.compute()
        except:
            auroc = torch.tensor(0.0)
            aupr = torch.tensor(0.0)

        return dict(auroc=auroc, aupr=aupr)

    def on_validation_epoch_end(self, outputs):
        """Compute final metrics after all validation batches"""
        # Compute accumulated metrics
        auroc_value = self.auroc_metric.compute()
        aupr_value = self.aupr_metric.compute()

        # Update outputs with actual metric values
        outputs['auroc'] = auroc_value.item()
        outputs['aupr'] = aupr_value.item()

        # Call parent method to log and update history
        super().on_validation_epoch_end(outputs)

        # Reset metrics for next epoch
        self.auroc_metric.reset()
        self.aupr_metric.reset()

    def configure_optimizers(self):
        """Default optimizer configuration - can be overridden"""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        return dict(optimizer=optimizer, scheduler=scheduler)

    def configure_early_stoppers(self):
        """Default early stopper configuration - can be overridden"""
        return dict(
            train=EarlyStopper(patience=10, mode='min', min_delta=0.001, monitor='loss'),
            valid=EarlyStopper(patience=10, mode='max', min_delta=0.001, monitor='auroc',
                target_value=0.95)
        )
