import pytorch_lightning as pl
import torch
from collections import OrderedDict


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
