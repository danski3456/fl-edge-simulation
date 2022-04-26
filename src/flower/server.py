import torch
import flwr as fl
import pytorch_lightning as pl
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.models.base import MetricsCallback
from src.models.map import name_to_model
from src.datasets.map import name_to_dataset
from config import settings as st
from src.path_utils import save_metrics, save_model
from src.policies.base import get_fl_rounds


def get_on_fit_config_fn() -> Callable[[int], Dict[str, str]]:
    """Return a function which returns training configurations."""

    def fit_config(rnd: int) -> Dict[str, str]:
        """Return a configuration with static batch size and (local) epochs."""
        config = {
            "learning_rate": str(0.001),
            "batch_size": str(32),
            "round": rnd - 1,
        }
        return config

    return fit_config


def start_server(*args) -> None:
    # Define strategy

    model = name_to_model[st.MODEL_NAME]
    # metrics = MetricsCallback()

    # Start Flower server for three rounds of federated learning

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1,
        min_fit_clients=st.NUM_CLIENTS - 1,
        min_available_clients=st.NUM_CLIENTS - 1,
        on_fit_config_fn=get_on_fit_config_fn(),
        eval_fn=get_eval_fn(model),
    )

    num_fl_rounds = len(get_fl_rounds())

    history = fl.server.start_server(
        server_address="[::]:8080",
        config={"num_rounds": num_fl_rounds},
        strategy=strategy,
    )

    save_metrics(history.metrics_centralized, "server")


def get_eval_fn(model):
    """Return an evaluation function for server-side evaluation."""
    dataset = name_to_dataset[st.DATASET_NAME]
    dataloader = dataset.load_dataloader(stage="server_eval")

    # The `evaluate` function will be called after every round
    def evaluate(
        weights: fl.common.Weights,
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:

        save_model(weights, st.MODEL_NAME, st.DATASET_NAME)
        model.set_parameters(weights)
        trainer = pl.Trainer(accelerator="auto", devices="auto")
        results = trainer.test(model, dataloader)
        loss = results[-1]["test_loss"]

        preds = []
        targets = []
        for batch_idx, batch in enumerate(dataloader):
            targets.append(batch[1])
            preds.append(model.predict_step(batch, batch_idx))
        # %%
        preds, targets = torch.cat(preds), torch.cat(targets)
        f1s = model._get_f1s(preds, targets)

        return loss, {**results[0], **f1s}

    return evaluate


if __name__ == "__main__":
    start_server()
