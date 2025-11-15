import torch
import torch.nn as nn
from torchmetrics import Accuracy


class Classifier(nn.Module):
    def __init__(self, encoder, num_classes, device=None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = encoder.to(self.device)
        self.loss_fn = nn.CrossEntropyLoss()
        self.acc_metric = Accuracy(task="multiclass", num_classes=num_classes).to(self.device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

    def train_step(self, batch):
        images = batch["image"].to(self.device)
        labels = batch["label"].to(self.device)

        self.optimizer.zero_grad()
        logits = self.encoder(images)
        loss = self.loss_fn(logits, labels)
        loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            acc = self.acc_metric(logits, labels)
        return dict(loss=loss, acc=acc)

    @torch.no_grad()
    def eval_step(self, batch):
        images = batch["image"].to(self.device)
        labels = batch["label"].to(self.device)

        logits = self.encoder(images)
        loss = self.loss_fn(logits, labels)
        acc = self.acc_metric(logits, labels)
        return dict(loss=loss, acc=acc)
    
    @torch.no_grad()
    def pred_step(self, batch):
        images = batch["image"].to(self.device)
        labels = batch["label"]
        logits = self.encoder(images)
        preds = torch.softmax(logits, dim=1).argmax(dim=1)
        return dict(images=images, labels=labels, preds=preds)