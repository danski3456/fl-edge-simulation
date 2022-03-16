# %%
import matplotlib.pyplot as plt
import json
from numpy import save
import pandas as pd
import seaborn as sns

from config import settings as st
from src.path_utils import load_metrics, save_image

# %%

server_metrics = load_metrics("server")

# %%
client_metrics = []
for cl in range(st.NUM_CLIENTS):
    cl_m = load_metrics(f"client_{cl}")
    df = pd.DataFrame(cl_m).T.applymap(lambda x: list(x.values())[0])
    df = df.reset_index()
    df.columns = ["round", "Train", "Validation"]
    df["client_id"] = cl
    client_metrics.append(df)
# %%

df = pd.concat(client_metrics)
df = df.melt(
    id_vars=["client_id", "round"],
    value_vars=["Train", "Validation"],
    value_name="Loss",
)

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
