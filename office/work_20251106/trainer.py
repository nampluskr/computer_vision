import numpy as np
import torch
from tqdm import tqdm


def train(model, dataloader, optimizer):
    model.train()
    results = {}
    total = 0

    with tqdm(dataloader, desc="Train", leave=False, ascii=True) as progress_bar:
        for batch in progress_bar:
            batch_size = batch["image"].shape[0]
            total += batch_size

            outputs = model.train_step(batch, optimizer)
            for name, value in outputs.items():
                results.setdefault(name, 0.0)
                results[name] += value.item() * batch_size

            progress_bar.set_postfix({name: f"{value / total:.3f}" 
                for name, value in results.items()})

    return {name: value / total for name, value in results.items()}


def evaluate(model, dataloader):
    model.eval()
    results = {}
    total = 0

    with tqdm(dataloader, desc="Eval", leave=False, ascii=True) as progress_bar:
        for batch in progress_bar:
            batch_size = batch["image"].shape[0]
            total += batch_size

            outputs = model.eval_step(batch)
            for name, value in outputs.items():
                results.setdefault(name, 0.0)
                results[name] += value.item() * batch_size

            progress_bar.set_postfix({name: f"{value / total:.3f}" 
                for name, value in results.items()})

    return {name: value / total for name, value in results.items()}


def predict(model, dataloader):
    model.eval()
    results = {}
    total = 0

    with tqdm(dataloader, desc="Predict", leave=False, ascii=True) as progress_bar:
        for batch in progress_bar:
            batch_size = batch["image"].shape[0]
            total += batch_size

            outputs = model.pred_step(batch)

            for name, value in outputs.items():
                if torch.is_tensor(value):
                    if value.dim() == 4 and value.shape[1] in [1, 3]:
                        value = value.cpu().permute(0, 2, 3, 1).numpy()
                    else:
                        value = value.cpu().numpy()
                elif isinstance(value, np.ndarray):
                    pass
                else:   # scalar or list
                    value = np.array(value)

                results.setdefault(name, [])
                results[name].append(value)

    results_np = {}
    for name, outputs_list in results.items():
        if isinstance(outputs_list[0], np.ndarray):
            results_np[name] = np.concatenate(outputs_list, axis=0)
        else:
            results_np[name] = np.array(outputs_list)

    return results_np


def fit(model, train_loader, optimizer, num_epochs, valid_loader=None):
    history = {"train": {}, "valid": {}}
    for epoch in range(1, num_epochs + 1):
        epoch_info = f"[{epoch:3d}/{num_epochs}]"
        train_results = train(model, train_loader, optimizer)
        train_info = ", ".join([f"{k}:{v:.3f}" for k, v in train_results.items()])

        for name, value in train_results.items():
            history["train"].setdefault(name, [])
            history["train"][name].append(value)

        if valid_loader is not None:
            valid_results = evaluate(model, valid_loader)
            valid_info = ", ".join([f"{k}:{v:.3f}" for k, v in valid_results.items()])

            for name, value in valid_results.items():
                history["valid"].setdefault(name, [])
                history["valid"][name].append(value)
            print(f"{epoch_info} {train_info} | (val) {valid_info}")
        else:
            print(f"{epoch_info} {train_info}")

    return history


class Trainer:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer

    def fit(self, train_loader, num_epochs, valid_loader=None):
        return fit(self.model, train_loader, self.optimizer, num_epochs, valid_loader=valid_loader)

    def evaluate(self, dataloader):
        return evaluate(self.model, dataloader)

    def predict(self, datalodaer):
        return perdict(self.model, dataloader)
