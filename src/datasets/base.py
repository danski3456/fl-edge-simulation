from abc import ABC
import random
from collections import defaultdict
from config import settings as st
from torch.utils.data import Dataset, DataLoader
from src.path_utils import get_path
from typing import List, Dict
from pathlib import Path


class Dataset(ABC):
    """
    Abstract class to interface with datasets
    """

    def __init__(self, name):
        self.name = name
        self.path = get_path(st.RAW_DATASETS) / name
        self.path.mkdir(parents=True, exist_ok=True)

    def download_dataset(self) -> None:
        pass

    def load_dataset(self, train: bool = True) -> Dataset:
        pass

    def load_dataloader(self, train: bool = True, batch_size: int = 32) -> DataLoader:
        pass

    def client_loader(
        self,
        round_samples: Dict[str, List[int]],
        val_fraction: float = 0.2,
        seed: int = 42,
        **kwargs,
    ) -> Dict[str, Dict[int, DataLoader]]:

        loaders_per_round = defaultdict(dict)
        dataset = self.load_dataset()
        for round, samples in round_samples.items():

            random.Random(seed).shuffle(samples)
            split_index = int(len(samples) * (1 - val_fraction))
            train_samples, val_samples = samples[:split_index], samples[split_index:]

            loaders_per_round[int(round)] = {
                "train": DataLoader(dataset, sampler=train_samples, **kwargs),
                "val": DataLoader(dataset, sampler=val_samples, **kwargs),
            }

        return loaders_per_round
