# %%
import json
import numpy as np
from collections import defaultdict
import torch
import pytorch_lightning as pl
from config import settings as st
from src.datasets.map import name_to_dataset
from src.models.map import name_to_model
from src.models.base import MetricsCallback
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from src.path_utils import final_assignmnet_path, save_metrics, save_model


# %%

if __name__ == "__main__":

    # %%

    model = name_to_model[st.MODEL_NAME]
    dataset = name_to_dataset[st.DATASET_NAME]
    metrics = MetricsCallback()

    assignment_path = final_assignmnet_path()
    with open(assignment_path, "r") as fh:
        assignment_order = json.load(fh)

    idx_samples = defaultdict(list)
    for k, v in assignment_order.items():
        for round, items in v.items():
            idx_samples[round].extend(items)
    # %%

    loaders = dataset.client_loader(idx_samples)
    # %%
    trainer = pl.Trainer(max_epochs=1, callbacks=[metrics])
    results = []
    for round, loader in loaders.items():
        result = trainer.fit(model, loader["train"], loader["val"])
        metrics.persist_round(round)
        results.append(result)
    # %%
    save_metrics(metrics.metrics, f"central_server")

    # %%
    weights = model.get_parameters()
    save_model(weights, st.MODEL_NAME, st.DATASET_NAME, fl=False)

# %%
