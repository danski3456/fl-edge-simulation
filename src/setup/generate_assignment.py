# %%
import pandas as pd
import numpy as np
from config import settings as st
from collections import defaultdict
from src.datasets.map import name_to_dataset
from torch.utils.data import Dataset
from src.path_utils import original_assignment_path
from src.setup.optimal_sharing import solve_instance

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
    C = df["class"].nunique()
    N = st.NUM_CLIENTS

    sampling_weight = {}
    for n in range(N):
        sampling_weight[n] = np.ones(C)
    if C > N:
        for c in range(C):
            sampling_weight[c % N][c] += st.IID_SCORE
    else:
        for n in range(N):
            sampling_weight[n][n % C] += st.IID_SCORE

    for n in range(N):
        sampling_weight[n] = sampling_weight[n] / (sampling_weight[n].sum())
        sampling_weight[n] = dict((i, v) for i, v in enumerate(sampling_weight[n]))

    # print(sampling_weight)

    # if st.IID is False:
    #     if C > N:
    #         for c in range(C):
    #             class_to_client[c % N].append(c)
    #     else:
    #         for cl in range(N):
    #             class_to_client[cl] = cl % C
    # else:
    #     for cl in range(N):
    #         class_to_client[cl] = list(range(C))

    samples_all_clients = []
    for cl in range(st.NUM_CLIENTS):
        # df_ = df[df["class"].isin(class_to_client[cl])]
        df_ = df.copy()
        df_["weight"] = df_["class"].map(sampling_weight[cl])
        client_samples = df_.sample(
            samples_per_client[cl], replace=True, weights="weight"
        )

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
    dataset = name_to_dataset[st.DATASET_NAME].load_dataset(stage="train")
    df_assignment = generate_distribution(dataset)

    # %%

    if st.OPTIMAL_DATA_SHARING["topology"] != "none":

        class_count = (
            df_assignment.groupby(["client_id", "class"])["class"].size().to_dict()
        )
        for n in range(st.NUM_CLIENTS):
            for c in range(st.NUM_CLASSES):
                if (n, c) not in class_count:
                    class_count[(n, c)] = 0

        optimization_results = solve_instance(class_count)

        shared_rows = []
        for src, dest, cls, amount in optimization_results["exchanges"]:
            amount = int(amount)
            shared_samples = (
                df_assignment[
                    (df_assignment["client_id"] == src)
                    & (df_assignment["class"] == cls)
                ]
                .sample(amount)
                .copy()
            )
            shared_samples["client_id"] = dest
            shared_samples = shared_samples.reset_index(drop=True)
            shared_rows.append(shared_samples)

        df_assignment = pd.concat([df_assignment] + shared_rows)

    if not st.OPTIMAL_DATA_SHARING["multi-timeslot"]:
        print("Collapsing to one timeslot")
        df_assignment["arrival_time"] = 0

    # %%
    df_assignment.to_csv(path, index=False)
