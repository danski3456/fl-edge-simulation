import pytorch_lightning as pl
import torch
import copy
from collections import OrderedDict
from pytorch_lightning import Callback


class MetricsCallback(Callback):
    """PyTorch Lightning metric callback."""

    def __init__(self):
        super().__init__()
        self.metrics = {}
        self.cur_round = []

    def _get_logged_metric(self, trainer, pl_module):
        metrics = {}
        for key, values in trainer.logged_metrics.items():
            metrics[key] = values.item() if isinstance(values, torch.Tensor) else values
        return metrics

    def on_train_epoch_end(self, trainer, pl_module):
        metric = self._get_logged_metric(trainer, pl_module)
        self.cur_round.append(metric)

    def on_validation_epoch_end(self, trainer, pl_module):
        metric = self._get_logged_metric(trainer, pl_module)
        self.cur_round.append(metric)

    def persist_round(self, round):
        self.metrics[round] = copy.deepcopy(self.cur_round)
        self.cur_round = []


class Base(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.model_parts = []

    def _get_model_parts(self) -> list:
        """
        Returns a list of all the model parts
        """
        pass

    def get_parameters(self):
        """
        Extracts the model parameters as a long numpy list
        """
        parameters = []
        for parts in self._get_model_parts():
            parameters += [val.cpu().numpy() for _, val in parts.state_dict().items()]
        return parameters

    def set_parameters(self, parameters):

        model_parts = self._get_model_parts()
        idx = 0
        for parts in model_parts:
            keys = parts.state_dict().keys()
            len_keys = len(keys)
            local_params = parameters[idx : idx + len_keys]  # Keep used parameters

            idx += len_keys  # Move to the next parameter

            params_dict = zip(keys, local_params)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            parts.load_state_dict(state_dict, strict=True)
