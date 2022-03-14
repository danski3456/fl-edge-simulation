# %%
import json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import deque
from queue import Queue

from config import settings as st

from src.path_utils import original_assignment_path, final_assignmnet_path
from src.policies.map import name_to_policy

# %%

if __name__ == "__main__":

    path = original_assignment_path()
    df = pd.read_csv(path, index_col=None)

    policies = []
    for cl in range(st.NUM_CLIENTS):
        p = name_to_policy[st.POLICIES[cl]](cl, df.columns)
        policies.append(p)


    # The exact datapoints available to clientson each fl round
    fl_round_datapoints = dict((i, dict()) for i in range(st.NUM_CLIENTS))

    # %%
    fl_round = 0
    for t in range(st.TIMESLOTS):
        is_fl_round = t in st.FL_ROUNDS
        for cl in range(st.NUM_CLIENTS):

            sub_df = df[(df["arrival_time"] == t) & (df["client_id"] == cl)]
            
            fl_datapoints, share_dict = policies[cl].get_samples(sub_df, is_fl_round)
            fl_round_datapoints[cl][fl_round] = fl_datapoints

        if t in st.FL_ROUNDS:
            fl_round += 1

    # %%
    output_path = final_assignmnet_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as fh:
        json.dump(fl_round_datapoints, fh, indent=2)
    # %%
