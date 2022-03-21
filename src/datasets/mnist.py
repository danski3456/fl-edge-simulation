import numpy as np
from pathlib import Path
from random import shuffle
from torchvision.datasets import MNIST
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

import src.datasets.base as dbase


path = Path("/workspace/data/mnist")


class MNISTDataset(dbase.Dataset):

    name = "mnist"

    def __init__(self):
        super().__init__(self.name)

    def download_dataset(self) -> None:

        for label, is_train in zip(["train", "test"], [True, False]):
            MNIST(
                self.path / label,
                train=is_train,
                download=True,
                transform=transforms.ToTensor(),
            )

        return None

    def load_dataset(self, stage: str = "train", val_fraction: float = 0.2) -> Dataset:

        mode = "train" if stage in ["train", "server_eval"] else "test"
        path_ = self.path / mode
        dataset = MNIST(path_, mode, download=False, transform=transforms.ToTensor())

        if mode == "train":

            D = len(dataset)
            eval_size = int(D * val_fraction)
            eval_indices = self._get_generator().choice(
                np.arange(D), size=eval_size, replace=False
            )
            train_indices = np.setdiff1d(np.arange(D), eval_indices)

            if stage == "train":
                indices = train_indices
            else:
                indices = eval_indices
            dataset = dbase.Subset(dataset, indices, dataset.targets[indices])

        return dataset

    def load_dataloader(self, stage: str = "train", batch_size: int = 32) -> DataLoader:

        dataset = self.load_dataset(stage)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return data_loader
