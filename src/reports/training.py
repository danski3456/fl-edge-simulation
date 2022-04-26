# %%
import matplotlib.pyplot as plt
import json
from numpy import save
import pandas as pd
import seaborn as sns
from collections import ChainMap

from config import settings as st
from src.path_utils import load_metrics, save_image

# %%

server_metrics = load_metrics("server")

# %%

client_metrics = []
for cl in range(st.NUM_CLIENTS):
    metrics = dict(
        (r, dict(ChainMap(*metric)))
        for r, metric in load_metrics(f"client_{cl}").items()
    )
    df = pd.DataFrame.from_records(metrics).T
    df = df.rename(
        columns={
            "val_acc": "Validation Accuracy",
            "val_loss": "Validation Loss",
            "train_loss": "Train Loss",
        }
    )
    #     df.columns = ["Validation Accuracy", "Validation Loss", "Train Loss"]
    df = df.rename_axis("round").reset_index()
    df["round"] = df["round"].astype(int)
    df["client_id"] = cl
    client_metrics.append(df)
df = pd.concat(client_metrics)
# %%


# df = pd.concat(client_metrics)
df = df.melt(
    id_vars=["client_id", "round"],
    value_vars=["Train Loss", "Validation Loss"],
    value_name="Loss",
)
df = df.sort_values("round")

# %%
g = sns.FacetGrid(df, hue="variable", col="client_id", col_wrap=3)
g.map(sns.lineplot, "round", "Loss")
g.add_legend()
save_image(g, "training_clients.png")
# sns.lineplot(data=df[df["client_id"] == 0], x="round", y="loss", hue="variable")
# %%

x, y = zip(*server_metrics["test_loss"])
fig, ax = plt.subplots()
ax.plot(x, y, label="Validation Loss")
ax.set_xlabel("# Round")
ax.set_ylabel("Loss")
ax.set_title("Server Validation")
ax.legend()
save_image(fig, "training_server.png")
# %%

fig, ax = plt.subplots()
for i in range(10):
    x, f1 = zip(*server_metrics[f"f1-{i}"])
    ax.plot(x, f1, label=f"Digit {i}")
ax.legend(ncol=5, bbox_to_anchor=(0.5, 1), loc="lower center")
ax.set_xlabel("Round")
ax.set_ylabel("F1 Score")
save_image(fig, "f1_server.png")

# %%
