"""" filename: trainer.py """

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from abc import ABC, abstractmethod
from tqdm import tqdm
from time import time
from copy import deepcopy
import os
import logging


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


class BaseTrainer(ABC):
    def __init__(self, model, loss_fn=None, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if model is None:
            raise ValueError("Model must be provided")
        self.model = model.to(self.device)
        self.loss_fn = loss_fn.to(self.device) if isinstance(loss_fn, nn.Module) else loss_fn

        self.current_epoch = 0
        self.global_step = 0
        self.training = True
        self.best_model_state = None

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

    def on_train_epoch_start(self, epoch):
        self.current_epoch = epoch
        self.model.train()
        self.training = True
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

    def on_fit_start(self, output_dir):
        self.output_dir = output_dir
        self.fit_start_time = time()
        self.history = {"train": {}, "valid": {}}
        self._setup_optimizers()
        self._setup_early_stoppers()
        self.logger = self._setup_logger(output_dir)
        self.logger.info(f"Starting training on device: {self.device}")
        self.logger.info(f"Model: {self.model.__class__.__name__}")
        self.logger.info("-" * 50)

    def on_fit_end(self):
        self.total_time = time() - self.fit_start_time
        self.logger.info("-" * 50)
        self.logger.info(f"Total training time: {self._format_time(self.total_time)}")

        if self.best_model_state is not None and self.output_dir is not None:
            best_model_path = os.path.join(self.output_dir, f"{self.run_name}_epoch-{self.epoch}.pth")
            torch.save(self.best_model_state, best_model_path)
            self.logger.info(f"Best model saved to: {best_model_path}")

        self.logger.info("Training completed!")

    def _update_history(self, outputs):
        history_key = 'train' if self.training else 'valid'
        for key, value in outputs.items():
            self.history[history_key].setdefault(key, [])
            self.history[history_key][key].append(value)

    def _setup_logger(self, output_dir):
        logger = logging.getLogger(f"Trainer_{id(self)}")
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            console_formatter = logging.Formatter('%(message)s')
            file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

            console_handler = logging.StreamHandler()
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)

            if output_dir is not None:
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                file_handler = logging.FileHandler(os.path.join(output_dir, "training.log"))
                file_handler.setFormatter(file_formatter)
                logger.addHandler(file_handler)

        return logger

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
            self.train_early_stopper = early_stopper_config
            self.valid_early_stopper = None

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

    def _check_train_stopping(self, train_outputs):
        if self.train_early_stopper is None:
            return False

        monitor_key = self.train_early_stopper.monitor
        if monitor_key not in train_outputs:
            self.logger.error(f"Monitor key '{monitor_key}' not found in train outputs."
                              f" Available: {list(train_outputs.keys())}")
            return False

        score = train_outputs[monitor_key]
        if self.train_early_stopper(score):
            if self.train_early_stopper.target_reached:
                self.logger.info(f"Target train {monitor_key} reached: {score:.3f}")
            else:
                self.logger.info(f"Early stopping - Best train {monitor_key}: {self.train_early_stopper.best_score:.3f}")
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
        is_best = False
        if self.valid_early_stopper.best_score is None:
            is_best = True
        elif self.valid_early_stopper.mode == 'max':
            is_best = score > self.valid_early_stopper.best_score
        else:
            is_best = score < self.valid_early_stopper.best_score

        if is_best:
            self.best_model_state = deepcopy(self.model.state_dict())

        if self.valid_early_stopper(score):
            if self.valid_early_stopper.target_reached:
                self.logger.info(f"Target valid {monitor_key} reached: {score:.3f}")
            else:
                self.logger.info(f"Early stopping - Best valid {monitor_key}: {self.valid_early_stopper.best_score:.3f}")
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
                    name: f"{total_value / total_images:.3f}"
                    for name, total_value in accumulated_outputs.items()
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
        self.run_name = run_name or "best_model"
        self.on_fit_start(output_dir)

        for epoch in range(1, num_epochs + 1):
            self.epoch = epoch
            self.on_train_epoch_start(epoch)
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

            if self.scheduler is not None:
                self.scheduler.step()

        self.on_fit_end()

        return self.history

    def save_checkpoint(self, filepath):
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "global_step": self.global_step,
            "current_epoch": self.current_epoch,
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
        self.global_step = checkpoint["global_step"]
        self.current_epoch = checkpoint["current_epoch"]
        if "optimizer_state_dict" in checkpoint and self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.logger.info(f"Checkpoint loaded: {filepath}")

    def save_model(self, filepath):
        torch.save(self.model.state_dict(), filepath)
        self.logger.info(f"Model saved: {filepath}")

    def load_model(self, filepath, strict=True):
        self.model.load_state_dict(torch.load(filepath, map_location=self.device), strict=strict)
        self.logger.info(f"Model loaded: {filepath}")