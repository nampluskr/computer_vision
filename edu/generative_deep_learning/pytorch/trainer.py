import numpy as np
import torch
from tqdm import tqdm


def train(model, dataloader, optimizer):
    model.train()
    results = {}
    total_images = 0

    for batch in dataloader:
        batch_size = batch["image"].shape[0]
        total_images += batch_size

        outputs = model.train_step(batch, optimizer)
        for name, value in outputs.items():
            results.setdefault(name, 0.0)
            results[name] += value.item() * batch_size

    return {name: value / total_images for name, value in results.items()}


def evaluate(model, dataloader):
    model.eval()
    results = {}
    total_images = 0

    for batch in dataloader:
        batch_size = batch["image"].shape[0]
        total_images += batch_size

        outputs = model.eval_step(batch)
        for name, value in outputs.items():
            results.setdefault(name, 0.0)
            results[name] += value.item() * batch_size

    return {name: value / total_images for name, value in results.items()}


def predict(model, dataloader):
    model.eval()
    results_dict = {}

    for batch in dataloader:
        outputs = model.pred_step(batch)  # {"images": ..., "labels": ..., "preds": ..., }

        for name, value in outputs.items():
            if torch.is_tensor(value):
                if value.dim() == 4 and value.shape[1] in [1, 3]:   # (B, C, H, W)
                    value = value.cpu().permute(0, 2, 3, 1).numpy() # (B, H, W, C)
                else:
                    value = value.cpu().numpy()
            elif isinstance(value, np.ndarray):
                pass
            else:
                value = np.array(value)  # scalar or list

            results_dict.setdefault(name, [])
            results_dict[name].append(value)

    results = {}
    for name, value_list in results_dict.items():
        if isinstance(value_list[0], np.ndarray):
            results[name] = np.concatenate(value_list, axis=0)
        else:
            results[name] = np.array(value_list)

    return results


def fit(model, train_loader, optimizer, num_epochs, valid_loader=None):
    history = {"tarin": {}, "valid": {}}
    for epoch in range(1, num_epochs + 1):
        epoch_info = f"[{epoch:3d}/{num_epochs}]"
        train_results = train(model, train_loader, optimizer)
        train_info = ", ".join([f"{k}:{v:.3f}" for k, v in train_results.items()])

        for key, value in train_results.items():
            history["tarin"].setdefault(key, [])
            history["tarin"][key].append(value)

        if valid_loader is not None:
            valid_results = evaluate(model, valid_loader)
            valid_info = ", ".join([f"{k}:{v:.3f}" for k, v in valid_results.items()])

            for key, value in train_results.items():
                history["valid"].setdefault(key, [])
                history["valid"][key].append(value)
            print(f"{epoch_info} {train_info} | (val) {valid_info}")
        else:
            print(f"{epoch_info} {train_info}")

    return history
