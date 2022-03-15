import json
from config import settings as st
from pathlib import Path


def get_path(folder: str) -> Path:

    root_path = Path(st.ROOT_DIR)
    path = root_path / folder
    return path


def original_assignment_path():
    path = Path(st.ROOT_DIR)
    path /= st.ORIGINAL_ASSIGNMENT_DIR
    path /= st.DATASET_NAME
    path /= st.ORIGINAL_ASSIGNMENT_FILENAME
    return path


def final_assignmnet_path():
    path = Path(st.ROOT_DIR)
    path /= st.FINAL_ASSIGNMENT_DIR
    path /= st.DATASET_NAME
    path /= st.FINAL_ASSIGNMENT_FILENAME
    return path


def metrics_path(agent_id):
    path = Path(st.ROOT_DIR)
    path /= st.METRICS_DIR
    path /= str(agent_id)
    path /= st.METRICS_FILENAME
    return path


def save_metrics(metrics, agent_id):
    path = metrics_path(agent_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(metrics, fh, indent=2)


def load_metrics(agent_id):
    path = metrics_path(agent_id)
    with open(path, "r") as fh:
        return json.load(fh)


def save_image(fig, img_name):
    path = Path(st.ROOT_DIR)
    path /= st.IMAGE_DIR
    path /= st.DATASET_NAME
    path /= img_name
    print(path)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
