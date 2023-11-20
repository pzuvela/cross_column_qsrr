import os
from qsrr.tests.tests_common.constants import Dataset


class DatasetPaths:

    DATA_PATH: str = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), Dataset.DATA)
    SMRT_DATASET: str = os.path.join(DATA_PATH, "2023-11-18-smrt_dataset.csv")
    SMRT_DATASET_SMILES: str = os.path.join(DATA_PATH, "2023-11-18-smrt_dataset_smiles.csv")
