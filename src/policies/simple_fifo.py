from config import settings as st
import pandas as pd
from src.policies.base import Policy
from typing import List, Dict, Union


class SimpleFifoPolicy(Policy):
    def __init__(self, client_id: int, columns: List[str]):
        super().__init__(client_id, columns)
        self.num_share = 5


    def _compute_samples_to_share(self, samples: pd.DataFrame):
        samples = super()._compute_samples_to_share(samples)

        NG = self._get_neighbours()
        share_dict = dict()
        # import pdb; pdb.set_trace()
        for ng in NG:
            df_ = samples[samples["client_id"] == self.client_id]  # only share local
            df_ = df_.groupby("class", group_keys=False).apply(
                lambda x: x.sample(self.num_share, replace=True)
            )
            df_ = df_.drop_duplicates()
            share_dict[ng] = df_
        return share_dict

    def _compute_policy(
        self, new_samples: pd.DataFrame, received_samples: List[pd.DataFrame]
    ):
        samples = super()._compute_policy(new_samples, received_samples)
        cache = self.cache

        N = samples.shape[0]
        TC = self.cache_capacity  # How many new items can be added
        CU = cache.shape[0]

        if N + CU <= TC:
            num_store = N
            num_drop = 0
        elif (N + CU > TC) and (N <= TC):
            num_store = N
            num_drop = TC - N
        else:
            num_store = TC
            num_drop = TC

        self.most_recent_store_idx = samples.index[
            :num_store
        ]  # keep as many as possible

        local_cache = cache[cache["client_id"] == self.client_id]
        external_cache = cache[cache["client_id"] != self.client_id]
        if external_cache.shape[0] >= local_cache.shape[0]:
            drop_idx = cache.index[:num_drop]
        else:
            drop_idx = local_cache.index[:num_drop]

        self.most_recent_drop_idx = drop_idx


        # import pdb; pdb.set_trace()

        # NG = self._get_neighbours()
        # share_dict = dict()
        # import pdb; pdb.set_trace()
        # samples_to_share = pd.concat(samples, self.cache)
        # for ng in NG:
        #     df_ = samples[samples["client_id"] == self.client_id]  # only share local
        #     df_ = df_.groupby("class", group_keys=False).apply(
        #         lambda x: x.sample(self.num_share, replace=True)
        #     )
        #     df_ = df_.drop_duplicates()
        #     share_dict[ng] = df_

        self.most_recent_share_dict = self._compute_samples_to_share(samples)# share_dict

        return samples
