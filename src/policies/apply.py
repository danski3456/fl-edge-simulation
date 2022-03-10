# %%
import json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import deque
from queue import Queue

from config import settings as st

from src.path_utils import original_assignment_path, final_assignmnet_path
from src.policies.cache_capacities import get_cache_capacities

# %%

if __name__ == "__main__":

    path = original_assignment_path()
    df = pd.read_csv(path, index_col=None)

    capacities = get_cache_capacities()

    # Used by clients to store their data across time-slots
    client_caches = [deque(maxlen=capacities[i]) for i in range(st.NUM_CLIENTS)]

    # The exact datapoints available to clientson each fl round
    fl_round_datapoints = dict((i, dict()) for i in range(st.NUM_CLIENTS))

    # %%
    fl_round = 0
    for t in range(st.TIMESLOTS):
        for cl in range(st.NUM_CLIENTS):

            sub_df = df[(df["arrival_time"] == t) & (df["client_id"] == cl)]
            new_arrivals = sub_df["item_id"].values

            if t in st.FL_ROUNDS:

                fl_round_datapoints[cl][fl_round] = list(client_caches[cl]) + list(
                    new_arrivals
                )
                fl_round_datapoints[cl][fl_round] = [
                    int(x) for x in fl_round_datapoints[cl][fl_round]
                ]

            [client_caches[cl].append(it) for it in new_arrivals]

        if t in st.FL_ROUNDS:
            fl_round += 1

    # %%
    output_path = final_assignmnet_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as fh:
        json.dump(fl_round_datapoints, fh, indent=2)
    # %%
