import flwr as fl
import pytorch_lightning as pl
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.models.base import MetricsCallback
from src.models.map import name_to_model
from src.datasets.map import name_to_dataset
from config import settings as st
from src.path_utils import save_metrics, save_model


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
        min_fit_clients=st.NUM_CLIENTS,
        min_available_clients=st.NUM_CLIENTS,
        on_fit_config_fn=get_on_fit_config_fn(),
        eval_fn=get_eval_fn(model),
    )

    history = fl.server.start_server(
        server_address="[::]:8080",
        config={"num_rounds": len(st.FL_ROUNDS)},
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
        trainer = pl.Trainer(progress_bar_refresh_rate=0)
        results = trainer.test(model, dataloader)
        loss = results[-1]["test_loss"]
        # model.set_weights(weights)  # Update model with the latest parameters
        # loss, accuracy = model.evaluate(x_val, y_val)
        return loss, results[0]

    return evaluate


if __name__ == "__main__":
    start_server()
