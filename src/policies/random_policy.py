import numpy as np

from config import settings as st
import pandas as pd
from src.policies.base import Policy
from typing import List, Dict, Union


class RandomPolicy(Policy):
    def __init__(self, client_id: int, columns: List[str]):
        super().__init__(client_id, columns)
        self.num_share = 0.1


    def _compute_samples_to_share(self, samples: pd.DataFrame):
        samples = super()._compute_samples_to_share(samples)

        NG = self._get_neighbours()
        share_dict = dict()
        for ng in NG:
            df_ = samples[samples["client_id"] == self.client_id]  # only share local
            df_ = df_.groupby("class", group_keys=False).apply(
                lambda x: x.sample(frac=self.num_share, replace=True)
            )
            df_ = df_.drop_duplicates()
            share_dict[ng] = df_
        return share_dict

    def _compute_policy(
        self, new_samples: pd.DataFrame, received_samples: List[pd.DataFrame]
    ):
        samples = super()._compute_policy(new_samples, received_samples)
        samples["new"] = True
        cache = self.cache.copy()
        cache["new"] = False

        new_cache = pd.concat([cache, samples], axis=0)

        TC = self.cache_capacity  # How many new items can be added
        CC = cache.shape[0]

        drop = CC - TC if CC > TC else 0
        drop_selected = new_cache.sample(drop, replace=False)
        # drop_idx = np.random.choice(new_cache.index, size=(CC - TC))
        # else:
        #     drop_idx = np.array([])

        self.most_recent_drop_idx = drop_selected[drop_selected["new"] == False].index
        self.most_recent_store_idx = drop_selected[drop_selected["new"] == True].index
        # self.most_recent_store_idx = new_cache[] np.setdiff1d(samples.index, drop_idx)
        # self.most_recent_drop_idx = np.intersect1d(cache.index, drop_idx)


        self.most_recent_share_dict = self._compute_samples_to_share(samples) 

        return samples
