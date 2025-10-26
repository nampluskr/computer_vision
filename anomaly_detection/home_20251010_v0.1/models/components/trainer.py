import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from time import time

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix

import matplotlib
matplotlib.use('Agg')  # Use 'Agg' backend for non-GUI environments


class EarlyStopper:
    def __init__(self, patience=10, min_delta=1e-3, mode='max', target_value=None):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.target_value = target_value

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
            elif self.mode == 'min' and score <= self.target_value:
                self.target_reached = True
                self.early_stop = True
                return True

        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == 'max':
            if score > self.best_score + self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
        else:  # mode == 'min'
            if score < self.best_score - self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True
            return True

        return False


class BaseTrainer:
    def __init__(self, model=None, optimizer=None, loss_fn=None, metrics=None, device=None,
                 scheduler=None, early_stopper_loss=None, early_stopper_auroc=None):

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if model is None:
            raise ValueError("Model must be provided or defined in the subclass __init__")
        self.model = model.to(self.device)

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.early_stopper_loss = early_stopper_loss    # mode = 'min'
        self.early_stopper_auroc = early_stopper_auroc  # mode = 'max'
        self.loss_fn = loss_fn.to(self.device) if isinstance(loss_fn, nn.Module) else loss_fn
        self.metrics = {}
        if metrics:
            for name, metric_fn in metrics.items():
                self.metrics[name] = metric_fn.to(self.device) if isinstance(metric_fn, nn.Module) else metric_fn

        self.eval_period = 5            # period for evaluation of classification metrics
        self.history = {}               # training history

        # Temperary variables for training/validation process
        self.fit_start_time = None      # time at the start of training
        self.epoch_start_time = None    # time at the start of each epoch
        self.epoch = None               # current epoch number
        self.num_epochs = None          # total number of epochs for training
        self.train_info = None          # string for training info
        self.valid_info = None          # string for validation info

    #############################################################
    # Hooks for training process
    #############################################################

    def on_train_start(self, train_loader):
        self.model.train()

    @torch.enable_grad()
    def train_step(self, batch):
        raise NotImplementedError

    def train_epoch(self, train_loader):
        results = {"loss": 0.0, **{name: 0.0 for name in self.metrics}}
        total = 0
        with tqdm(train_loader, desc=" > Training", leave=False, ascii=True) as pbar:
            for batch in pbar:
                batch_size = batch["image"].size(0)
                total += batch_size
                step_results = self.train_step(batch)

                for name, value in step_results.items():
                    results.setdefault(name, 0.0)
                    results[name] += value * batch_size

                pbar.set_postfix({name: f"{value/total:.3f}" for name, value in results.items()})
        return {name: value/total for name, value in results.items()}

    def on_train_end(self, train_results):
        self.train_info = ", ".join([f'{k}={v:.3f}' for k, v in train_results.items()])
        self.valid_info = None

        for name, value in train_results.items():
            self.history.setdefault(name, [])
            self.history[name].append(value)

        if self.early_stopper_loss is not None:
            loss = train_results.get('loss', float('inf'))
            if self.early_stopper_loss(loss):
                if self.early_stopper_loss.target_reached:
                    print(f"\n > Target loss reached! Loss: {loss:.3f} <= {self.early_stopper_loss.target_value:.3f}")
                else:
                    print(f"\n > Early stopping triggered! Best loss: {self.early_stopper_loss.best_score:.3f}")

    #############################################################
    # Hooks for validation process
    #############################################################

    def on_validation_start(self, valid_loader):
        self.model.eval()

    @torch.no_grad()
    def validation_step(self, batch):
        images = batch["image"].to(self.device)
        predictions = self.model(images)
        scores = predictions['pred_score'].cpu()
        if scores.ndim > 1:
            scores = scores.view(scores.size(0))
        return scores

    @torch.no_grad()
    def validation_epoch(self, loader):
        all_scores, all_labels = [], []
        with tqdm(loader, desc=" > Validation", leave=False, ascii=True) as pbar:
            for batch in loader:
                scores = self.validation_step(batch)
                labels = batch["label"].cpu()

                all_scores.append(scores)
                all_labels.append(labels)

        scores = torch.cat(all_scores)   # shape = [N_images]
        labels = torch.cat(all_labels)   # shape = [N_images]
        results = {}
        results["auroc"] = roc_auc_score(labels, scores)
        results["aupr"] = average_precision_score(labels, scores)
        return results, scores, labels

    def on_validation_end(self, valid_results, scores, labels):
        self.valid_info = ", ".join([f'{k}={v:.3f}' for k, v in valid_results.items()])
        elapsed_time = time() - self.epoch_start_time
        epoch_info = f"[{self.epoch:3d}/{self.num_epochs}]"
        print(f" {epoch_info} {self.train_info} | {self.valid_info} ({elapsed_time:.1f}s)")

        for name, value in valid_results.items():
            self.history.setdefault(name, [])
            self.history[name].append(value)

        if self.early_stopper_auroc is not None:
            auroc = valid_results.get('auroc', 0.0)
            if self.early_stopper_auroc(auroc):
                if self.early_stopper_auroc.target_reached:
                    print(f"\n > Target AUROC reached! AUROC: {auroc:.3f} >= {self.early_stopper_auroc.target_value:.3f}")
                else:
                    print(f"\n > Early stopping triggered! Best AUROC: {self.early_stopper_auroc.best_score:.3f}")

        if self.epoch % self.eval_period == 0:
            method = "f1"
            threshold = compute_threshold(scores, labels, method=method)
            eval_results = evaluate_classification(scores, labels, threshold)
            eval_info1 = ", ".join([f"{k}={v:.3f}" for k, v in eval_results.items() if isinstance(v, float)])
            eval_info2 = ", ".join([f"{k.upper()}={v}" for k, v in eval_results.items() if isinstance(v, int)])
            print(f" > {eval_info1} | {eval_info2}\n")

    #############################################################
    # Hooks for fitting process
    #############################################################

    def on_fit_start(self ):
        print("\n > Start training...\n")
        self.fit_start_time = time()

    def on_fit_end(self, weight_path=None):
        elapsed_time = time() - self.fit_start_time
        hours, reminder = divmod(int(elapsed_time), 3600)
        minutes, seconds = divmod(reminder, 60)
        print(f"\n > Training finished... in {hours:02d}:{minutes:02d}:{seconds:02d}\n")
        self.save_model(weight_path)

    #############################################################
    # Hooks for epoch process
    #############################################################

    def on_epoch_start(self):
        self.epoch_start_time = time()

    def on_epoch_end(self):
        if self.valid_info is None:
            elapsed_time = time() - self.epoch_start_time
            epoch_info = f" [{self.epoch:3d}/{self.num_epochs}]"
            print(f" {epoch_info} {self.train_info} ({elapsed_time:.1f}s)")

        if self.scheduler is not None:
            old_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step()
            new_lr = self.optimizer.param_groups[0]['lr']

            # if abs(old_lr - new_lr) > 1e-7:
            #     print(f" > Learning rate changed: {old_lr:.3e} -> {new_lr:.3e}\n")

            if old_lr > 0 and abs(old_lr - new_lr) / old_lr > 0.001:
                print(f" > Learning rate changed: {old_lr:.6e} -> {new_lr:.6e}\n")

    #############################################################
    # Main fit function
    #############################################################

    def fit(self, train_loader, num_epochs, valid_loader=None, weight_path=None):
        self.num_epochs = num_epochs
        self.on_fit_start()

        for epoch in range(1, num_epochs + 1):
            self.epoch = epoch
            self.on_epoch_start()

            self.on_train_start(train_loader)
            train_results = self.train_epoch(train_loader)
            self.on_train_end(train_results)

            if self.early_stopper_loss is not None and self.early_stopper_loss.early_stop:
                break

            if valid_loader:
                self.on_validation_start(valid_loader)
                valid_results, scores, labels = self.validation_epoch(valid_loader)
                self.on_validation_end(valid_results, scores, labels)

                if self.early_stopper_auroc is not None and self.early_stopper_auroc.early_stop:
                    break

            self.on_epoch_end()

        self.on_fit_end(weight_path)
        return self.history

    #############################################################
    # Model save / load
    #############################################################

    def save_model(self, weight_path):
        if weight_path is not None:
            result_dir = os.path.abspath(os.path.dirname(weight_path))
            os.makedirs(result_dir, exist_ok=True)
            checkpoint = {"model": self.model.state_dict()}

            if self.optimizer is not None:
                checkpoint["optimizer"] = self.optimizer.state_dict()
            if self.scheduler is not None:
                checkpoint["scheduler"] = self.scheduler.state_dict()
            torch.save(checkpoint, weight_path)
            print(f" > Model weights saved to: {weight_path}\n")

    def load_model(self, weight_path):
        if os.path.isfile(weight_path):
            checkpoint = torch.load(weight_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model"])
            print(f" > Model weights loaded from: {weight_path}")

            if self.optimizer is not None and "optimizer" in checkpoint:
                self.optimizer.load_state_dict(checkpoint["optimizer"])
                print(" > Optimizer state loaded.")
            if self.scheduler is not None and "scheduler" in checkpoint:
                self.scheduler.load_state_dict(checkpoint["scheduler"])
                print(" > Scheduler state loaded.")
        else:
            print(f" > No model weights found at: {weight_path}\n")

    #############################################################
    # Model test (saving anomaly maps)
    #############################################################

    @torch.no_grad()
    def save_maps(self, test_loader, result_dir=None,  desc=None, show_image=False,
            skip_normal=False, skip_anomaly=False, num_max=-1, normalize=True):

        if result_dir is not None:
            os.makedirs(result_dir, exist_ok=True)

        self.model.eval()
        num_saved = 0
        for batch in test_loader:
            labels = batch["label"].cpu().numpy()
            images = batch["image"].to(self.device)
            defect_types = batch["defect_type"]
            categories = batch["category"]
            masks = batch["mask"].cpu().numpy() if "mask" in batch else None
            has_mask = batch["has_mask"][0].item() if "has_mask" in batch else True

            prediction = self.model(images)
            anomaly_maps = prediction["anomaly_map"].cpu().numpy()
            scores = prediction["pred_score"].cpu().numpy()

            for i in range(images.size(0)):
                label = int(labels[i])
                score = float(scores[i])
                defect_type = defect_types[i]
                category = categories[i]

                is_normal = (label == 0)
                is_anomaly = (label == 1)
                if skip_normal and is_normal: continue
                if skip_anomaly and is_anomaly: continue
                if num_max > 0 and num_saved >= num_max: continue
                num_saved += 1

                img_tensor = images[i].cpu()
                original = denormalize(img_tensor).clamp(0, 1) if normalize else img_tensor.clamp(0, 1)
                amap = anomaly_maps[i]
                anomaly_map = (amap - amap.min()) / (amap.max() - amap.min() + 1e-8)

                if has_mask:
                    binary_mask = None
                    if masks is not None:
                        mask = masks[i]
                        if mask.ndim == 3 and mask.shape[0] == 1:
                            mask = mask[0]
                        binary_mask = mask

                    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
                    titles = [f"{category}: {defect_type}", "Mask", f"Anomaly Score: {score:.4f}"]
                    images_vis = [original, binary_mask, anomaly_map]
                    cmaps = [None, "gray", "jet"]
                else:
                    fig, axes = plt.subplots(1, 2, figsize=(9, 3))
                    titles = [f"{category}: {defect_type}", f"Anomaly Score: {score:.4f}"]
                    images_vis = [original, anomaly_map]
                    cmaps = [None, "jet"]

                for ax, img, title, cmap in zip(axes, images_vis, titles, cmaps):
                    ax.imshow(check_shape(img), cmap=cmap)
                    ax.set_title(title)
                    ax.axis("off")

                fig.tight_layout()
                if result_dir is not None:
                    label_name = "normal" if is_normal else "anomaly"
                    filename = f"image_{desc + '_' if desc else ''}{label_name}_{num_saved:04d}.png"
                    filepath = os.path.join(result_dir, filename)
                    fig.savefig(filepath, dpi=150)

                if show_image: plt.show()
                plt.close(fig)

            if num_max > 0 and num_saved >= num_max:
                break

        if result_dir is not None and num_saved > 0:
            print(f" > Saved {num_saved} anomaly maps to {result_dir}")

    @torch.no_grad()
    def save_histogram(self, test_loader, result_dir=None, desc=None, show_image=False,):
        import seaborn as sns

        if result_dir is not None:
            os.makedirs(result_dir, exist_ok=True)

        self.model.eval()
        all_scores = []
        all_labels = []

        for batch in test_loader:
            labels = batch["label"].cpu().numpy()
            images = batch["image"].to(self.device)

            prediction = self.model(images)
            scores = prediction["pred_score"].cpu().numpy()

            all_scores.append(scores)
            all_labels.append(labels)

        all_scores = np.concatenate(all_scores).ravel()
        all_labels = np.concatenate(all_labels).ravel()
        normal_scores = all_scores[all_labels == 0]
        anomaly_scores = all_scores[all_labels == 1]

        thresholds = {}
        thresholds['f1'] = compute_threshold(all_scores, all_labels, method="f1")
        thresholds['f1_uniform'] = compute_threshold(all_scores, all_labels, method="f1_uniform")
        thresholds['roc'] = compute_threshold(all_scores, all_labels, method="roc")
        thresholds['percentile'] = compute_threshold(all_scores, all_labels, method="percentile")

        fig, ax = plt.subplots(figsize=(12, 6))
        bins = 50

        if len(normal_scores) > 0:
            sns.histplot(normal_scores, bins=bins, kde=True, color='blue',
                         label=f'Normal (n={len(normal_scores)})',
                         alpha=0.6, edgecolor='black', ax=ax)
        if len(anomaly_scores) > 0:
            sns.histplot(anomaly_scores, bins=bins, kde=True, color='red',
                         label=f'Anomaly (n={len(anomaly_scores)})',
                         alpha=0.6, edgecolor='black', ax=ax)

        colors = {'f1': 'green', 'f1_uniform': 'lime', 'roc': 'purple', 'percentile': 'orange'}
        linestyles = {'f1': '-', 'f1_uniform': ':', 'roc': '--', 'percentile': '-.'}

        for method, threshold in thresholds.items():
            label_name = method.replace('_', ' ').upper()
            ax.axvline(threshold, color=colors[method], linestyle=linestyles[method],
                    linewidth=2, label=f'{label_name}: {threshold:.4f}')

        ax.set_xlabel('Anomaly Score', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Distribution of Anomaly Scores with Thresholds', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        if result_dir is not None:
            prefix = f"{desc}_" if desc else ""
            filename = f"histogram_{prefix}scores.png"
            filepath = os.path.join(result_dir, filename)
            fig.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f" > Saved histogram to {filepath}")

        if show_image: plt.show()
        plt.close(fig)

    @torch.no_grad()
    def save_results(self, test_loader, result_dir=None, desc=None):
        if result_dir is not None:
            os.makedirs(result_dir, exist_ok=True)

        self.model.eval()
        all_scores = []
        all_labels = []

        for batch in test_loader:
            labels = batch["label"].cpu().numpy()
            images = batch["image"].to(self.device)

            prediction = self.model(images)
            scores = prediction["pred_score"].cpu().numpy()

            all_scores.append(scores)
            all_labels.append(labels)

        # Concatenate all batches
        all_scores = np.concatenate(all_scores).ravel()
        all_labels = np.concatenate(all_labels).ravel()

        # Separate normal and anomaly scores
        normal_scores = all_scores[all_labels == 0]
        anomaly_scores = all_scores[all_labels == 1]

        # Compute AUROC and AUPR
        auroc = roc_auc_score(all_labels, all_scores)
        aupr = average_precision_score(all_labels, all_scores)

        # Compute thresholds using different methods
        thresholds = {}
        thresholds['f1'] = compute_threshold(all_scores, all_labels, method="f1")
        thresholds['f1_uniform'] = compute_threshold(all_scores, all_labels, method="f1_uniform")
        thresholds['roc'] = compute_threshold(all_scores, all_labels, method="roc")
        thresholds['percentile'] = compute_threshold(all_scores, all_labels, method="percentile")

        if result_dir is not None:
            prefix = f"{desc}_" if desc else ""
            txt_filename = f"results_{prefix}thresholds.txt"
            txt_filepath = os.path.join(result_dir, txt_filename)

            # Write results to file
            with open(txt_filepath, 'w') as f:
                f.write("="*70 + "\n")
                f.write("ANOMALY DETECTION RESULTS\n")
                f.write("="*70 + "\n\n")

                # Overall Performance Metrics (AUROC & AUPR)
                f.write("Overall Performance Metrics\n")
                f.write("-"*70 + "\n")
                f.write(f"{'AUROC (ROC-AUC)':25s}: {auroc:12.6f} ({auroc*100:6.2f}%)\n")
                f.write(f"{'AUPR (AP)':25s}: {aupr:12.6f} ({aupr*100:6.2f}%)\n")
                f.write("\n\n")

                # Threshold values
                f.write("Threshold Values\n")
                f.write("-"*70 + "\n")
                f.write(f"{'Method':25s}  {'Threshold':>12s}\n")
                f.write("-"*70 + "\n")
                f.write(f"{'F1 (Percentile)':25s}: {thresholds['f1']:12.6f}\n")
                f.write(f"{'F1 (Uniform)':25s}: {thresholds['f1_uniform']:12.6f}\n")
                f.write(f"{'ROC (Youden J)':25s}: {thresholds['roc']:12.6f}\n")
                f.write(f"{'Percentile (95%)':25s}: {thresholds['percentile']:12.6f}\n")
                f.write("\n\n")

                # Classification results for each method
                f.write("Classification Results by Threshold Method\n")
                f.write("="*70 + "\n\n")

                for method_name, threshold in thresholds.items():
                    # Compute predictions
                    predictions = (all_scores >= threshold).astype(int)

                    # Compute metrics
                    acc = accuracy_score(all_labels, predictions)
                    prec = precision_score(all_labels, predictions, zero_division=0)
                    rec = recall_score(all_labels, predictions, zero_division=0)
                    f1 = f1_score(all_labels, predictions, zero_division=0)

                    # Confusion matrix
                    tn, fp, fn, tp = confusion_matrix(all_labels, predictions).ravel()

                    # Write method results
                    method_display = method_name.replace('_', ' ').upper()
                    f.write(f"Method: {method_display}\n")
                    f.write(f"Threshold: {threshold:.6f}\n")
                    f.write("-"*70 + "\n")

                    # Metrics
                    f.write(f"{'Accuracy':20s}: {acc:8.4f} ({acc*100:6.2f}%)\n")
                    f.write(f"{'Precision':20s}: {prec:8.4f} ({prec*100:6.2f}%)\n")
                    f.write(f"{'Recall (TPR)':20s}: {rec:8.4f} ({rec*100:6.2f}%)\n")
                    f.write(f"{'F1 Score':20s}: {f1:8.4f}\n")
                    f.write("\n")

                    # Confusion Matrix
                    f.write("Confusion Matrix:\n")
                    f.write(f"                 Predicted Normal  Predicted Anomaly\n")
                    f.write(f"Actual Normal    {tn:16d}  {fp:17d}\n")
                    f.write(f"Actual Anomaly   {fn:16d}  {tp:17d}\n")
                    f.write("\n")

                    # Additional metrics
                    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
                    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

                    f.write(f"{'True Positives  (TP)':20s}: {tp:8d}\n")
                    f.write(f"{'False Negatives (FN)':20s}: {fn:8d}\n")
                    f.write(f"{'True Negatives  (TN)':20s}: {tn:8d}\n")
                    f.write(f"{'False Positives (FP)':20s}: {fp:8d}\n")
                    f.write(f"{'Specificity (TNR)':20s}: {specificity:8.4f}\n")
                    f.write(f"{'False Positive Rate':20s}: {fpr:8.4f}\n")
                    f.write(f"{'False Negative Rate':20s}: {fnr:8.4f}\n")
                    f.write("\n" + "="*70 + "\n\n")

                # Score statistics
                f.write("Score Statistics\n")
                f.write("="*70 + "\n")
                f.write(f"Normal Samples (n={len(normal_scores)})\n")
                f.write("-"*70 + "\n")
                f.write(f"  {'Mean':10s}: {normal_scores.mean():10.6f}\n")
                f.write(f"  {'Std':10s}: {normal_scores.std():10.6f}\n")
                f.write(f"  {'Min':10s}: {normal_scores.min():10.6f}\n")
                f.write(f"  {'Max':10s}: {normal_scores.max():10.6f}\n")
                f.write(f"  {'Median':10s}: {np.median(normal_scores):10.6f}\n")
                f.write(f"  {'Q1':10s}: {np.percentile(normal_scores, 25):10.6f}\n")
                f.write(f"  {'Q3':10s}: {np.percentile(normal_scores, 75):10.6f}\n")
                f.write("\n")

                f.write(f"Anomaly Samples (n={len(anomaly_scores)})\n")
                f.write("-"*70 + "\n")
                f.write(f"  {'Mean':10s}: {anomaly_scores.mean():10.6f}\n")
                f.write(f"  {'Std':10s}: {anomaly_scores.std():10.6f}\n")
                f.write(f"  {'Min':10s}: {anomaly_scores.min():10.6f}\n")
                f.write(f"  {'Max':10s}: {anomaly_scores.max():10.6f}\n")
                f.write(f"  {'Median':10s}: {np.median(anomaly_scores):10.6f}\n")
                f.write(f"  {'Q1':10s}: {np.percentile(anomaly_scores, 25):10.6f}\n")
                f.write(f"  {'Q3':10s}: {np.percentile(anomaly_scores, 75):10.6f}\n")
                f.write("\n")

                # Separation metrics
                f.write("Separation Metrics\n")
                f.write("-"*70 + "\n")
                mean_diff = anomaly_scores.mean() - normal_scores.mean()
                f.write(f"{'Mean Difference':25s}: {mean_diff:10.6f}\n")

                # Cohen's d (effect size)
                pooled_std = np.sqrt(((len(normal_scores) - 1) * normal_scores.std()**2 +
                                    (len(anomaly_scores) - 1) * anomaly_scores.std()**2) /
                                    (len(normal_scores) + len(anomaly_scores) - 2))
                cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0.0

                cohens_label = "Cohen's d (effect size)"
                f.write(f"{cohens_label:25s}: {cohens_d:10.6f}\n")

                # Overlap
                overlap_count = np.sum((normal_scores > anomaly_scores.min()) &
                                    (normal_scores < anomaly_scores.max()))
                overlap_pct = overlap_count / len(normal_scores) * 100 if len(normal_scores) > 0 else 0.0
                f.write(f"{'Score Overlap':25s}: {overlap_pct:9.2f}%\n")
                f.write("\n" + "="*70 + "\n")

            print(f" > Saved results to {txt_filepath}")


#############################################################
# Helper functions
#############################################################

def compute_threshold(scores, labels, method="f1", percentile=95):
    if isinstance(scores, torch.Tensor):
        scores = scores.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    if method == "f1":
        # Percentile-based: More samples where scores are concentrated
        thresholds = np.percentile(scores, np.linspace(1, 99.9, 200))
        best_f1, best_threshold = 0.0, 0.5

        for thr in thresholds:
            preds = (scores >= thr).astype(int)
            f1 = f1_score(labels, preds, zero_division=0)
            if f1 > best_f1:
                best_f1, best_threshold = f1, thr

        return float(best_threshold)

    elif method == "f1_uniform":
        # Uniform sampling: Equal intervals across score range
        thresholds = np.linspace(scores.min(), scores.max(), 1000)
        f1_scores = []

        for threshold in thresholds:
            predictions = (scores >= threshold).astype(int)
            f1 = f1_score(labels, predictions, zero_division=0)
            f1_scores.append(f1)

        best_idx = np.argmax(f1_scores)
        return float(thresholds[best_idx])

    elif method == "roc":
        # Use Youden's J statistic (maximize TPR - FPR)
        fpr, tpr, thresholds = roc_curve(labels, scores)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        return thresholds[best_idx]

    elif method == "percentile":
        # Use 95th percentile of normal scores
        normal_scores = scores[labels == 0]
        if len(normal_scores) == 0:
            return scores.mean()
        return np.percentile(normal_scores, percentile)

    else:
        raise ValueError(f"Unknown method: {method}. Choose from 'f1', 'roc', 'percentile'")


def evaluate_classification(scores, labels, threshold):
    preds = (scores >= threshold).long()
    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    return {
        "th": float(threshold),
        "acc": (tp + tn) / (tp + tn + fp + fn + 1e-8),
        "prec": tp / (tp + fp + 1e-8),
        "recall": tp / (tp + fn + 1e-8),
        "f1": 2 * tp / (2 * tp + fp + fn + 1e-8),
        "tp": tp, "fn": fn, "tn": tn, "fp": fp,
    }


def denormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(tensor.device)
    return tensor * std + mean


def check_shape(img):
    if torch.is_tensor(img):
        img = img.detach().cpu().numpy()
    if img.ndim == 2:
        return img
    if img.ndim == 3:
        if img.shape[0] == 1:
            return img[0]
        elif img.shape[0] == 3:
            return np.transpose(img, (1, 2, 0))
        elif img.shape[2] == 3:
            return img
        else:
            raise ValueError(f"Invalid shape for 3D array: {img.shape}")
    raise ValueError(f"Unsupported image shape: {img.shape}")
