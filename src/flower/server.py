import flwr as fl
import pytorch_lightning as pl
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.models.map import name_to_model
from src.datasets.map import name_to_dataset
from config import settings as st


def start_server() -> None:
    # Define strategy

    model = name_to_model[st.MODEL_NAME]

    strategy = fl.server.strategy.FedAvg(
        eval_fn=get_eval_fn(model),
    )

    # Start Flower server for three rounds of federated learning
    fl.server.start_server(
        server_address="[::]:8080",
        config={"num_rounds": len(st.FL_ROUNDS)},
        strategy=strategy,
    )


def get_eval_fn(model):
    """Return an evaluation function for server-side evaluation."""
    dataset = name_to_dataset[st.DATASET_NAME]
    dataloader = dataset.load_dataloader()

    # The `evaluate` function will be called after every round
    def evaluate(
        weights: fl.common.Weights,
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:

        model.set_parameters(weights)
        trainer = pl.Trainer(progress_bar_refresh_rate=0)
        results = trainer.test(model, dataloader)
        loss = results[0]["test_loss"]
        # model.set_weights(weights)  # Update model with the latest parameters
        # loss, accuracy = model.evaluate(x_val, y_val)
        return loss, {"loss": loss}

    return evaluate


if __name__ == "__main__":
    start_server()
