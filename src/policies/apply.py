# %%
import json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import deque, defaultdict
from queue import Queue
from copy import deepcopy

from config import settings as st

from src.path_utils import original_assignment_path, final_assignmnet_path
from src.policies.map import name_to_policy
from src.policies.base import get_fl_rounds

# %%

if __name__ == "__main__":

    FL_ROUNDS = get_fl_rounds()
    print(FL_ROUNDS)

    path = original_assignment_path()
    df = pd.read_csv(path, index_col=None)

    policies = []
    for cl in range(st.NUM_CLIENTS):
        policy_name = st.POLICIES[cl] if isinstance(st.POLICIES, list) else st.POLICIES
        p = name_to_policy[policy_name](cl, df.columns)
        policies.append(p)

    # The exact datapoints available to clientson each fl round
    fl_round_datapoints = dict((i, dict()) for i in range(st.NUM_CLIENTS))

    shared_datapoints_prev = dict((i, []) for i in range(st.NUM_CLIENTS))



    # %%
    fl_round = 0
    for t in range(st.TIMESLOTS):

        shared_datapoints_new = dict((i, []) for i in range(st.NUM_CLIENTS))
        is_fl_round = t in FL_ROUNDS

        for cl in range(st.NUM_CLIENTS):

            sub_df = df[(df["arrival_time"] == t) & (df["client_id"] == cl)]
            received_dp = shared_datapoints_prev[cl]

            fl_datapoints, share_dict = policies[cl].get_samples(
                sub_df, received_dp, t, is_fl_round
            )
            for dest_cl, dps in share_dict.items():
                shared_datapoints_new[dest_cl].append(dps)

            if is_fl_round:
                fl_round_datapoints[cl][fl_round] = fl_datapoints

        if is_fl_round:
            fl_round += 1

        shared_datapoints_prev = deepcopy(shared_datapoints_new)

    # %%
    output_path = final_assignmnet_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as fh:
        json.dump(fl_round_datapoints, fh, indent=2)
    # %%
