# %%
import numpy as np
import torch
import pytorch_lightning as pl
from config import settings as st
from src.datasets.map import name_to_dataset
from src.models.map import name_to_model
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from src.path_utils import load_model, save_image, save_metrics

# %%

if __name__ == "__main__":

    # %%

    for mode, label in [(True, "federated"), (False, "central")]:

        model = name_to_model[st.MODEL_NAME]
        dataset = name_to_dataset[st.DATASET_NAME]
        dataloader_test = dataset.load_dataloader(stage="test")

        model = load_model(model, st.MODEL_NAME, st.DATASET_NAME, mode)

        trainer = pl.Trainer()
        results = trainer.test(model, dataloaders=dataloader_test)

        # %%
        # %%
        # cfs = []
        preds = []
        targets = []
        for batch_idx, batch in enumerate(dataloader_test):
            targets.append(batch[1])
            preds.append(model.predict_step(batch, batch_idx))
        # %%
        preds, targets = torch.cat(preds), torch.cat(targets)

        # preds = np.hstack([x.numpy() for x in preds])
        # targets = np.hstack([x.numpy() for x in targets])

        # %%
        cf = model.confusion(preds, targets)
        # cfs.append(cf)

        # %%
        assert np.allclose(
            results[0]["test_acc"],
            (cf.diag().sum() / cf.sum()).item(),
        )
        # %%
        core_metric = dict(core_metric=model.core_metric(preds, targets).item())

        save_metrics(core_metric, f"{label}", train=False)

        # %%

        C = cf.shape[0]
        df_cm = pd.DataFrame(cf, index=range(C), columns=range(C))
        fig, ax = plt.subplots(figsize=(8, 4), facecolor=(1, 1, 1))
        g = sns.heatmap(df_cm, annot=True, fmt="g", cmap="Blues", ax=ax)
        fig.tight_layout()
        ax.set_title(f"{label} -- Accuracy: {results[0]['test_acc']:.2%}")
        save_image(fig, f"confusion_matrix_{label}.png")

# %%
