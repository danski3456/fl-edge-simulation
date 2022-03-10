from config import settings as st
from src.datasets.map import name_to_dataset

if __name__ == "__main__":
    name_to_dataset[st.DATASET_NAME].download_dataset()
