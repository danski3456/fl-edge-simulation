from pathlib import Path
from random import shuffle
from torchvision.datasets import MNIST
from torch.utils.data import Dataset, DataLoader
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

    def load_dataset(self, train: bool = True) -> Dataset:
        ext = "train" if train else "test"
        path_ = self.path / ext
        dataset = MNIST(path_, train, download=False, transform=transforms.ToTensor())
        return dataset

    def load_dataloader(self, train: bool = True, batch_size: int = 32) -> DataLoader:

        dataset = self.load_dataset(train)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return data_loader
