import os

class DatasetPaths:

    DATA_PATH: str = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), "data")
    SMRT_DATASET: str = os.path.join(DATA_PATH, "2023-11-18-smrt_dataset.csv")
    SMRT_DATASET_SMILES: str = os.path.join(DATA_PATH, "2023-11-18-smrt_dataset_smiles.csv")
