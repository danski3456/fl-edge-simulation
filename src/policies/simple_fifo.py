import pandas as pd
from src.policies.base import Policy
from typing import List, Dict, Union


class SimpleFifoPolicy(Policy):
    def __init__(self, client_id: int, columns: List[str]):
        super().__init__(client_id, columns)

    
    def _compute_policy(self, new_samples: pd.DataFrame):
        super()._compute_policy(new_samples)

        N = new_samples.shape[0]
        C = min(N, self.cache_capacity) # How many new items can be added

        self.most_recent_store_idx = new_samples.index[:C] # keep as many as possible
        self.most_recent_drop_idx = self.cache.index[:C] # drop the first C
        self.most_recent_share_dict = dict()
