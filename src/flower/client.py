import flwr as fl
from pathlib import Path
import sys
import pytorch_lightning as pl
from collections import OrderedDict

from config import settings as st

import json
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST


from src.datasets.map import name_to_dataset
from src.models.map import name_to_model
from src.path_utils import final_assignmnet_path


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, data_loaders):
        self.model = model
        self.time_slot = 0

        self.data_loaders = data_loaders

    def get_parameters(self):
        return self.model.get_parameters()

    def fit(self, parameters, config):
        self.model.set_parameters(parameters)

        train_loader = self.data_loaders[self.time_slot]["train"]
        val_loader = self.data_loaders[self.time_slot]["val"]
        trainer = pl.Trainer(max_epochs=1, progress_bar_refresh_rate=0)
        trainer.fit(self.model, train_loader, val_loader)

        self.time_slot += 1

        return self.get_parameters(), len(train_loader.sampler), {}

    def evaluate(self, parameters, config):
        pass


def start_client(client_id: str) -> None:
    # Model and data

    model = name_to_model[st.MODEL_NAME]

    assignment_path = final_assignmnet_path()
    with open(assignment_path, "r") as fh:
        assignment_order = json.load(fh)[client_id]

    data_loaders = name_to_dataset[st.DATASET_NAME].client_loader(
        assignment_order, batch_size=32
    )

    # Flower client
    client = FlowerClient(model, data_loaders)
    fl.client.start_numpy_client("0.0.0.0:8080", client)


if __name__ == "__main__":
    id_ = sys.argv[1]
    start_client(id_)
