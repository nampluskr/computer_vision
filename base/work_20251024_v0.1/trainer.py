"""" filename: trainer.py """

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from abc import ABC, abstractmethod
from tqdm import tqdm
from time import time
import os
import logging


class EarlyStopper:
    def __init__(self, patience=10, min_delta=1e-3, mode='max', target_value=None):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.target_value = target_value
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
    
    def on_train_epoch_end(self, outputs):
        metrics = {}
        for key in outputs[0].keys():
            if isinstance(outputs[0][key], torch.Tensor):
                metrics[key] = sum(o[key].item() for o in outputs) / len(outputs)
            else:
                metrics[key] = sum(o[key] for o in outputs) / len(outputs)
        
        self._update_history(outputs)
        
        epoch_info = f"[{self.current_epoch:3d}/{self.num_epochs}]"
        train_info = ", ".join([f"{k}:{v:.3f}" for k, v in metrics.items()])
        time_info = self._format_time(time() - self.epoch_start_time)
        self.logger.info(f"{epoch_info} {train_info} ({time_info})")
    
    def on_validation_epoch_start(self):
        self.model.eval()
        self.training = False
    
    def on_validation_epoch_end(self, outputs):
        """
        FIX #1: Validation epoch이 끝난 후 metrics를 정리하는 hook
        Subclass에서 필요시 override하여 epoch-level metric 계산 가능
        """
        metrics = {}
        for key in outputs[0].keys():
            if isinstance(outputs[0][key], torch.Tensor):
                metrics[key] = sum(o[key].item() for o in outputs) / len(outputs)
            else:
                metrics[key] = sum(o[key] for o in outputs) / len(outputs)
        
        self._update_history(outputs)
        valid_info = ", ".join([f"{k}:{v:.3f}" for k, v in metrics.items()])
        self.logger.info(f"{valid_info}")

    def on_train_batch_start(self, batch, batch_idx): pass
    def on_train_batch_end(self, outputs, batch, batch_idx): pass

    def on_validation_batch_start(self, batch, batch_idx): pass
    def on_validation_batch_end(self, outputs, batch, batch_idx): pass

    def on_fit_start(self, num_epochs, output_dir):
        self.num_epochs = num_epochs
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
        self.logger.info("Training completed!")
    
    def _update_history(self, outputs):
        history_key = 'train' if self.training else 'valid'
        
        for key in outputs[0].keys():
            if isinstance(outputs[0][key], torch.Tensor):
                avg_value = sum(o[key].item() for o in outputs) / len(outputs)
            else:
                avg_value = sum(o[key] for o in outputs) / len(outputs)
            
            if key not in self.history[history_key]:
                self.history[history_key][key] = []
            self.history[history_key][key].append(avg_value)
    
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
        
        # 기본적으로 'loss'를 사용, 없으면 첫 번째 스칼라 metric 사용
        score_key = 'loss' if 'loss' in train_outputs[0] else None
        if score_key is None:
            for key in train_outputs[0].keys():
                if isinstance(train_outputs[0][key], torch.Tensor) and train_outputs[0][key].numel() == 1:
                    score_key = key
                    break
        
        if score_key is None:
            return False
        
        avg_score = sum(o[score_key].item() if isinstance(o[score_key], torch.Tensor) else o[score_key] 
                       for o in train_outputs) / len(train_outputs)
        
        if self.train_early_stopper(avg_score):
            if self.train_early_stopper.target_reached:
                self.logger.info(f"\nTarget train {score_key} reached: {avg_score:.4f}")
            else:
                self.logger.info(f"\nEarly stopping - Best train {score_key}: {self.train_early_stopper.best_score:.4f}")
            return True
        return False
    
    def _check_valid_stopping(self, valid_outputs):
        """
        FIX #2: Best model checkpoint 자동 저장 기능 추가
        """
        if self.valid_early_stopper is None:
            return False
        
        # 기본적으로 'acc'를 사용, 없으면 'loss', 없으면 첫 번째 스칼라 metric 사용
        score_key = None
        if 'acc' in valid_outputs[0]:
            score_key = 'acc'
        elif 'auroc' in valid_outputs[0]:  # anomaly detection용 추가
            score_key = 'auroc'
        elif 'loss' in valid_outputs[0]:
            score_key = 'loss'
        else:
            for key in valid_outputs[0].keys():
                if isinstance(valid_outputs[0][key], torch.Tensor) and valid_outputs[0][key].numel() == 1:
                    score_key = key
                    break
        
        if score_key is None:
            return False
        
        avg_score = sum(o[score_key].item() if isinstance(o[score_key], torch.Tensor) else o[score_key] 
                       for o in valid_outputs) / len(valid_outputs)
        
        # Best model 체크 및 저장
        if self.output_dir is not None:
            is_best = False
            
            if self.valid_early_stopper.best_score is None:
                is_best = True
            elif self.valid_early_stopper.mode == 'max':
                is_best = avg_score > self.valid_early_stopper.best_score
            else:
                is_best = avg_score < self.valid_early_stopper.best_score
            
            if is_best:
                best_model_path = os.path.join(self.output_dir, "best_model.pth")
                self.save_model(best_model_path)
                self.logger.info(f"Best model saved (valid {score_key}: {avg_score:.4f})")
        
        if self.valid_early_stopper(avg_score):
            if self.valid_early_stopper.target_reached:
                self.logger.info(f"\nTarget valid {score_key} reached: {avg_score:.4f}")
            else:
                self.logger.info(f"\nEarly stopping - Best valid {score_key}: {self.valid_early_stopper.best_score:.4f}")
            return True
        return False
    
    @torch.enable_grad()
    def _train_epoch(self, train_loader):
        epoch_outputs = []
        total_samples = 0
        accumulated_metrics = {}
        
        epoch_info = f"[{self.current_epoch:3d}/{self.num_epochs}]"
        with tqdm(train_loader, desc=f"{epoch_info} Training", leave=False, ascii=True) as progress_bar:
            for batch_idx, batch in enumerate(progress_bar):
                batch_size = batch["image"].shape[0]
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
                
                epoch_outputs.append(outputs)
                total_samples += batch_size
                
                for key, value in outputs.items():
                    if isinstance(value, torch.Tensor):
                        value = value.item()
                    if key not in accumulated_metrics:
                        accumulated_metrics[key] = 0.0
                    accumulated_metrics[key] += value * batch_size
                
                progress_bar.set_postfix({
                    name: f"{total_value / total_samples:.3f}" 
                    for name, total_value in accumulated_metrics.items()
                })
                
                self.on_train_batch_end(outputs, batch, batch_idx)
        
        return epoch_outputs
    
    @torch.no_grad()
    def _validate_epoch(self, valid_loader):
        epoch_outputs = []
        total_samples = 0
        accumulated_metrics = {}
        
        epoch_info = f"[{self.current_epoch:3d}/{self.num_epochs}]"
        with tqdm(valid_loader, desc=f"{epoch_info} Validation", leave=False, ascii=True) as progress_bar:
            for batch_idx, batch in enumerate(progress_bar):
                batch_size = batch["image"].shape[0]
                self.on_validation_batch_start(batch, batch_idx)
                
                outputs = self.validation_step(batch, batch_idx)
                epoch_outputs.append(outputs)
                total_samples += batch_size
                
                for key, value in outputs.items():
                    if isinstance(value, torch.Tensor):
                        value = value.item()
                    if key not in accumulated_metrics:
                        accumulated_metrics[key] = 0.0
                    accumulated_metrics[key] += value * batch_size
                
                progress_bar.set_postfix({
                    name: f"{total_value / total_samples:.3f}" 
                    for name, total_value in accumulated_metrics.items()
                })
                
                self.on_validation_batch_end(outputs, batch, batch_idx)
        
        return epoch_outputs
    
    def fit(self, train_loader, num_epochs, valid_loader=None, output_dir=None):
        self.on_fit_start(num_epochs, output_dir)
        
        for epoch in range(1, num_epochs + 1):
            self.on_train_epoch_start(epoch)
            train_outputs = self._train_epoch(train_loader)
            self.on_train_epoch_end(train_outputs)
            
            if self._check_train_stopping(train_outputs):
                break
            
            if valid_loader is not None:
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
        if hasattr(self, 'logger'):
            self.logger.info(f"Checkpoint saved: {filepath}")
        else:
            print(f"Checkpoint saved: {filepath}")
    
    def load_checkpoint(self, filepath, strict=True):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
        self.global_step = checkpoint["global_step"]
        self.current_epoch = checkpoint["current_epoch"]
        if "optimizer_state_dict" in checkpoint and self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if hasattr(self, 'logger'):
            self.logger.info(f"Checkpoint loaded: {filepath}")
        else:
            print(f"Checkpoint loaded: {filepath}")
    
    def save_model(self, filepath):
        torch.save(self.model.state_dict(), filepath)
        if hasattr(self, 'logger'):
            self.logger.info(f"Model saved: {filepath}")
        else:
            print(f"Model saved: {filepath}")
    
    def load_model(self, filepath, strict=True):
        self.model.load_state_dict(torch.load(filepath, map_location=self.device), strict=strict)
        if hasattr(self, 'logger'):
            self.logger.info(f"Model loaded: {filepath}")
        else:
            print(f"Model loaded: {filepath}")