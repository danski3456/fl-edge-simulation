# %%
import pandas as pd
import numpy as np
from config import settings as st
from collections import defaultdict
from src.datasets.map import name_to_dataset
from torch.utils.data import Dataset
from src.path_utils import original_assignment_path

# %%


# %%


def build_arrival_rate():
    type_ = st.ARRIVAL_RATE["type"]
    num_clients = st.NUM_CLIENTS
    timeslots = st.TIMESLOTS

    if type_ == "constant":
        quantity = st.ARRIVAL_RATE["quantity"]
        arrival_rate = np.ones((timeslots, num_clients)).astype(int)
        arrival_rate *= quantity  # constant in this case

    else:
        raise NotImplementedError

    return arrival_rate


# %%


def generate_distribution(dataset: Dataset) -> pd.DataFrame:
    """
    Generates a dataframe with the information of which samples
    arrives at each client in which time-slot.
    """

    arrival_rate = build_arrival_rate()

    samples_per_client = arrival_rate.sum(axis=0)

    df = pd.DataFrame.from_records(
        ((i, t.item()) for i, t in enumerate(dataset.targets))
    )
    df.columns = ["item_id", "class"]

    class_to_client = defaultdict(list)
    if st.IID is False:
        C = df["class"].nunique()
        N = st.NUM_CLIENTS
        if C > N:
            for c in range(C):
                class_to_client[c % N].append(c)
        else:
            for cl in range(N):
                class_to_client[cl] = cl % C
    else:
        for cl in range(N):
            class_to_client[cl] = list(range(C))

    samples_all_clients = []
    for cl in range(st.NUM_CLIENTS):
        df_ = df[df["class"].isin(class_to_client[cl])]
        client_samples = df_.sample(samples_per_client[cl], replace=True)
        client_timeslots = sum(
            [[t] * ar for ar, t in zip(arrival_rate[:, 0], range(st.TIMESLOTS))], []
        )
        client_samples["arrival_time"] = client_timeslots
        client_samples["client_id"] = cl
        samples_all_clients.append(client_samples)

    samples_all_clients = pd.concat(samples_all_clients, axis=0)
    return samples_all_clients


# %%
if __name__ == "__main__":

    # %%

    path = original_assignment_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    # Load dataset
    dataset = name_to_dataset[st.DATASET_NAME].load_dataset()
    df_assignment = generate_distribution(dataset)

    # %%

    df_assignment.to_csv(path, index=False)

# %%
