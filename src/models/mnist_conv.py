"""Adapted from the PyTorch Lightning quickstart example.

Source: https://pytorchlightning.ai/ (2021/02/04)
"""

# %%

import numpy as np
import torch
import torchmetrics
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST
import pytorch_lightning as pl

from src.models.base import Base

# %%


class MNISTConvNet(Base):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # fully connected layer, output 10 classes
        self.out = nn.Linear(32 * 7 * 7, 10)
        self.core_metric = torchmetrics.F1Score(num_classes=10)
        self.accuracy = torchmetrics.Accuracy()
        self.confusion = torchmetrics.ConfusionMatrix(num_classes=10)

    def _get_model_parts(self):
        return [self.conv1, self.conv2, self.out]

    def _get_f1s(self, preds, targets) -> dict:
        cf = self.confusion(preds, targets)

        f1s = {}
        N = cf.sum().item()
        for i in range(10):

            try:
                TP = cf[i, i].item()
                FP = cf[i, :].sum().item() - TP
                FN = cf[:, i].sum().item() - TP
                C = np.delete(cf, i, 0)
                C = np.delete(C, i, 1)
                TN = C.sum().item()

                precision = TP / (TP + FP)
                recall = TP / (TP + FN)
                f1 = (2 * TP) / (2 * TP + FP + FN)
            except ZeroDivisionError:
                f1 = 0

            f1s[f"f1-{i}"] = f1

        return f1s

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return F.softmax(output, axis=1)  # return probability across classes

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def _eval(self, batch, stage=None):
        x, y = batch
        x_hat = self.conv1(x)
        x_hat = self.conv2(x_hat)
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x_hat = x_hat.view(x_hat.size(0), -1)
        output = self.out(x_hat)
        loss = F.cross_entropy(output, y)
        if stage == "train":
            return output, loss
        elif stage in ["val", "test"]:
            _, preds = torch.max(output, 1)
            acc = self.accuracy(preds, y)
            return output, preds, loss, acc

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = None):
        output, preds, loss, acc = self._eval(batch, "val")
        return preds

    def training_step(self, train_batch, batch_idx):
        output, loss = self._eval(train_batch, "train")
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        output, preds, loss, acc = self._eval(batch, "val")
        metrics = {"val_acc": acc, "val_loss": loss}
        self.log_dict(metrics)
        return metrics

    def test_step(self, batch, batch_idx):
        output, preds, loss, acc = self._eval(batch, "test")
        metrics = {"test_acc": acc, "test_loss": loss}
        self.log_dict(metrics)
        return metrics


# %%

if __name__ == "__main__":

    # %%
    model = MNISTConvNet()
    from src.datasets.mnist import MNISTDataset
    import pytorch_lightning as plt

    from src.models.base import MetricsCallback

    metrics = MetricsCallback()
    dataloader = MNISTDataset().load_dataloader(stage="train")
    trainer = pl.Trainer(
        accelerator="auto", devices="auto", max_epochs=1, callbacks=[metrics]
    )

    trainer.fit(model, dataloader)

    # %%

    # Test

    dataloader_test = MNISTDataset().load_dataloader(stage="test")
    results = trainer.test(model, dataloaders=dataloader_test)

# %%
