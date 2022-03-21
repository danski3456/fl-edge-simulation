from abc import ABC
import torch
import numpy as np
import random
from collections import defaultdict
from config import settings as st
from torch.utils.data import Dataset, DataLoader
from src.path_utils import get_path
from typing import List, Dict
from pathlib import Path


class Subset(Dataset):
    r"""
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
        labels(sequence) : targets as required for the indices. will be the same length as indices
    """

    def __init__(self, dataset, indices, labels):
        self.dataset = torch.utils.data.Subset(dataset, indices)
        self.targets = labels

    def __getitem__(self, idx):
        image = self.dataset[idx][0]
        target = self.targets[idx]
        return (image, target)

    def __len__(self):
        return len(self.targets)


class Dataset(ABC):
    """
    Abstract class to interface with datasets
    """

    def __init__(self, name):
        self.name = name
        self.path = get_path(st.RAW_DATASETS) / name
        self.path.mkdir(parents=True, exist_ok=True)
        self.seed = st.SEED

    def _get_generator(self):
        return np.random.RandomState(seed=self.seed)

    def download_dataset(self) -> None:
        pass

    def load_dataset(self, stage: str = "train") -> Dataset:
        pass

    def load_dataloader(self, stage: str = "train", batch_size: int = 32) -> DataLoader:
        pass

    def client_loader(
        self,
        round_samples: Dict[str, List[int]],
        val_fraction: float = 0.2,
        seed: int = 42,
        **kwargs,
    ) -> Dict[str, Dict[int, DataLoader]]:

        loaders_per_round = defaultdict(dict)
        dataset = self.load_dataset(stage="train")
        for round, samples in round_samples.items():

            random.Random(seed).shuffle(samples)
            split_index = int(len(samples) * (1 - val_fraction))
            train_samples, val_samples = samples[:split_index], samples[split_index:]

            loaders_per_round[int(round)] = {
                "train": DataLoader(dataset, sampler=train_samples, **kwargs),
                "val": DataLoader(dataset, sampler=val_samples, **kwargs),
            }

        return loaders_per_round
