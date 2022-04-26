import numpy as np
import json
from config import settings as st
from pathlib import Path


def get_path(folder: str) -> Path:

    root_path = Path(st.ROOT_DIR)
    path = root_path / folder
    return path


def original_assignment_path():
    path = Path(st.ROOT_DIR)
    path /= st.ASSETS_DIR
    path /= st.ORIGINAL_ASSIGNMENT_DIR
    path /= st.DATASET_NAME
    path /= st.ORIGINAL_ASSIGNMENT_FILENAME
    return path


def final_assignmnet_path():
    path = Path(st.ROOT_DIR)
    path /= st.ASSETS_DIR
    path /= st.FINAL_ASSIGNMENT_DIR
    path /= st.DATASET_NAME
    path /= st.FINAL_ASSIGNMENT_FILENAME
    return path


def metrics_path(agent_id, train=True):
    path = Path(st.ROOT_DIR)
    path /= st.ASSETS_DIR
    path /= st.METRICS_DIR
    path /= "train" if train else "test"
    path /= str(agent_id)
    path /= st.METRICS_FILENAME
    return path


def save_metrics(metrics, agent_id, train=True):
    path = metrics_path(agent_id, train)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(metrics, fh, indent=2)


def load_metrics(agent_id, train=True):
    path = metrics_path(agent_id, train)
    with open(path, "r") as fh:
        return json.load(fh)


def save_image(fig, img_name):
    path = Path(st.ROOT_DIR)
    path /= st.ASSETS_DIR
    path /= st.IMAGE_DIR
    path /= st.DATASET_NAME
    path /= img_name
    print(path)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")


def model_path(model_name, dataset_name, fl=True):
    kind = "fl" if fl is True else "central"
    path = Path(st.ROOT_DIR)
    path /= st.ASSETS_DIR
    path /= st.MODELS_DIR
    path.mkdir(parents=True, exist_ok=True)
    name = f"{model_name}_{dataset_name}_{kind}.npz"
    path /= name
    return path


def save_model(weights, model_name, dataset_name, fl=True):
    path = model_path(model_name, dataset_name, fl)
    np.savez(path, *weights)


def load_model(model, model_name, dataset_name, fl=True):
    path = model_path(model_name, dataset_name, fl)
    weights = np.load(path)
    weights = [weights[f] for f in weights.files]
    model.set_parameters(weights)
    return model
