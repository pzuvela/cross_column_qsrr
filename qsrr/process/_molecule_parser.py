import pandas as pd
from pandas import DataFrame

from rdkit import Chem


class MoleculeParser:

    def __init__(
        self,
        data_df: DataFrame
    ):
        self.__data_df: DataFrame = data_df

    def inchi2smiles(self) -> DataFrame:

        _smiles_list = []
        _rt_list = []

        for _col, _row in self.__data_df.iterrows():
            try:
                _smiles_list.append(
                    Chem.MolToSmiles(
                        Chem.MolFromInchi(
                            _row["inchi"]
                        )
                    )
                )
                _rt_list.append(_row['rt'])
            except Exception:  # noqa (need a broad exception here)
                print(f"Error while parsing inchi to SMILES...")

        return pd.DataFrame().from_dict(
            {'smiles': _smiles_list, 'rt': _rt_list}
        )
