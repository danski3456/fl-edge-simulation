from abc import ABC, abstractmethod
from collections import deque
from pdb import Pdb
import pdb
from typing import Union, List, Dict
import pandas as pd
from config import settings as st


def get_cache_capacities() -> List[int]:
    """
    Parses the parameter file and generates a list
    of the cache capacities of each player
    """
    param_type = st.CACHE_CAPACITY["type"]
    if param_type == "constant":
        quantity = st.CACHE_CAPACITY["quantity"]
        capacities = [quantity for _ in range(st.NUM_CLIENTS)]

    return capacities


def get_fl_rounds() -> List[int]:
    """
    Applies one of several functions to calculate
    fl rounds from time-slots
    """
    fl_type = st.FL_ROUNDS["type"]
    ts = list(range(st.TIMESLOTS))
    if fl_type == "all":
        rounds = ts
    elif fl_type == "sample":
        freq = st.FL_ROUNDS["freq"]
        rounds = ts[::freq]
    else:
        raise NotImplementedError
    return rounds


class Policy(ABC):
    """
    Abstact class representing a general policy
    to be implemented in the administriation of a local cache
    """

    def __init__(self, client_id: int, columns: List[str]):

        self.client_id = client_id
        self.cache_capacity = get_cache_capacities()[client_id]
        self.cache = pd.DataFrame(columns=columns)

        self.most_recent_store_idx = []
        self.most_recent_drop_idx = []
        self.most_recent_share_dict = dict()

    def _get_availability(self, t: int) -> int:

        if self.client_id in st.RESOURCES:
            resource_type = st.RESOURCES[self.client_id]
        else:
            resource_type = "full"
        if resource_type == "first-half":
            perc = 1.0 if t <= (st.TIMESLOTS / 2) else 0.0
        elif resource_type == "full":
            perc = 1.0

        availability = int(st.MAX_RESOURCES * perc)

        return availability

    def _get_neighbours(self):

        topology = st.TOPOLOGY
        N = st.NUM_CLIENTS
        cid = self.client_id
        if topology == "ring":
            neighbours = [((cid - 1) % N), ((cid + 1) % N)]
        elif topology == "none":
            neighbours = []
        else:
            raise NotImplementedError

        return neighbours

    def _compute_policy(
        self, new_samples: pd.DataFrame, received_samples: List[pd.DataFrame]
    ):

        """
        new samples dataframe example
        |item_id|class|arrival_time|client_id|
        |-------|-----|------------|---------|
        |  27366|    3|           0|        0|
        |  23152|    4|           0|        0|
        |  59450|    0|           0|        0|
        """
        return pd.concat(received_samples + [new_samples], axis=0)

    def _update_cache(
        self, new_samples: pd.DataFrame, received_samples: List[pd.DataFrame]
    ):

        samples = self._compute_policy(new_samples, received_samples)

        cache = self.cache
        self.old_cache = cache.copy()

        cache = cache.drop(self.most_recent_drop_idx)
        df_append = samples.loc[self.most_recent_store_idx]
        cache = pd.concat([cache, df_append], axis=0)

        # import pdb; pdb.set_trace()

        self.cache = cache

    def get_samples(
        self,
        new_samples: pd.DataFrame,
        received_samples: List[pd.DataFrame],
        round: int,
        fl_round: bool = False,
    ):

        self._update_cache(new_samples, received_samples)
        max_samples = self._get_availability(round)

        if fl_round is True:
            train_samples = pd.concat([new_samples, self.cache] + received_samples)
            train_samples = train_samples.drop_duplicates()
            train_samples = list(train_samples["item_id"].values)
        else:
            train_samples = []
            # train_samples = list(self.cache["item_id"].values)

        train_samples = [int(x) for x in train_samples][:max_samples]

        return train_samples, self.most_recent_share_dict
