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
from src.path_utils import load_model, save_image

# %%

if __name__ == "__main__":

    # %%

    for mode, label in [(True, "federated"), (False, "central")]:

        model = name_to_model[st.MODEL_NAME]
        dataset = name_to_dataset[st.DATASET_NAME]
        dataloader_test = dataset.load_dataloader(train=False)

        model = load_model(model, st.MODEL_NAME, st.DATASET_NAME, mode)

        trainer = pl.Trainer()
        results = trainer.test(model, dataloaders=dataloader_test)

        # %%
        # %%
        cfs = []
        for batch_idx, batch in enumerate(dataloader_test):
            target = batch[1]
            pred = model.predict_step(batch, batch_idx)

            cf = model.confusion(pred, target)
            cfs.append(cf)

        confusion_matrix = np.sum(cfs)

        # %%
        assert np.allclose(
            results[0]["test_acc"],
            (confusion_matrix.diag().sum() / confusion_matrix.sum()).item(),
        )
        # %%

        C = confusion_matrix.shape[0]
        df_cm = pd.DataFrame(confusion_matrix, index=range(C), columns=range(C))
        fig, ax = plt.subplots(figsize=(8, 4), facecolor=(1, 1, 1))
        g = sns.heatmap(df_cm, annot=True, fmt="g", cmap="Blues", ax=ax)
        fig.tight_layout()
        ax.set_title(f"{label} -- Accuracy: {results[0]['test_acc']:.2%}")
        save_image(fig, f"confusion_matrix_{label}.png")

# %%
