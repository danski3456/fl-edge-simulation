# %%
import matplotlib.pyplot as plt
import json
import pandas as pd
import math
import seaborn as sns
import numpy as np

from config import settings as st
from src.path_utils import final_assignmnet_path, save_image
from src.datasets.map import name_to_dataset
from collections import Counter

# %%
dataset = name_to_dataset[st.DATASET_NAME].load_dataset(stage="train")

# %%

with open(final_assignmnet_path(), "r") as fh:
    assignments = json.load(fh)
# %%

# fig, axes = plt.subplots(2, 3, figsize=(12, 4))
df_all = []
for cl in range(st.NUM_CLIENTS):

    cl_as = assignments[str(cl)]

    dfs = []
    for key in cl_as.keys():
        class_items = [dataset.targets[x].item() for x in cl_as[key]]
        df = pd.DataFrame(class_items, columns=["class"])
        df["round"] = int(key)
        dfs.append(df)

    df = pd.concat(dfs).reset_index(drop=True)
    df["client"] = int(cl)
    df_all.append(df)

df = pd.concat(df_all).reset_index(drop=True)


# %%
fig, axes = plt.subplots(
    (st.NUM_CLIENTS // 3) + 1, 3, figsize=(14, 10), sharex=True, sharey=True
)
unique_classes = sorted(df["class"].unique())


for i in range(st.NUM_CLIENTS):

    ax = axes[i // 3, i % 3]
    # i = 0
    df_ = df[(df["client"] == i)]
    rounds = sorted(df_["round"].unique())
    totals = np.array([df_["round"].value_counts().to_dict()[x] for x in rounds])
    cant_rounds = np.array([0] * len(rounds)).astype("float")

    for c in unique_classes:

        height = df_[(df_["class"] == c)]["round"].value_counts().to_dict()
        for x in rounds:
            if x not in height:
                height.update({x: 0})
        yy = np.array([height[x] for x in rounds]) / totals
        ax.bar(rounds, yy, width=1, bottom=cant_rounds, align="edge")
        cant_rounds += yy

    ax.set_ylim([0, 1])
    ax.set_xlim([0, len(rounds)])
    ax.grid(True)
    ax.set_title(f"Client {i}")

axes[0, 1].legend(range(10), loc="lower center", bbox_to_anchor=(0.5, 1), ncol=5)


save_image(fig, "class_proportions.png")

# %%
