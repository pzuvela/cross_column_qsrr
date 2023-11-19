from numpy import ndarray

import pandas as pd
from pandas import DataFrame

import pytest

from qsrr.enums import MoleculeEncodingType
from qsrr.process import (
    MoleculeEncoder,
    MoleculeParser
)
from qsrr.tests.tests_common.constants import DatasetPaths


class TestMoleculeEncoder:

    @pytest.mark.parametrize(
        "molecule_encoding_type",
        list(MoleculeEncodingType)
    )
    def test_molecule_encoder(
        self,
        molecule_encoding_type: MoleculeEncodingType
    ):

        """

        Test for encoding SMILES to integer & one-hot encoding format

        Parameters
        ----------
        molecule_encoding_type: MoleculeEncodingType, Integer or OneHot encoding

        Returns
        -------
        None

        """

        # Load Data
        _data_df: DataFrame = pd.read_csv(DatasetPaths.SMRT_DATASET_SMILES)

        # Convert InChi to SMILES
        _data_df: DataFrame = MoleculeParser(_data_df).inchi2smiles()

        # Build vocabulary of SMILES characters
        _vocabulary, _inverse_vocabulary = MoleculeEncoder.build(_data_df, file_path=None)

        # Encode SMILES to features
        _encoded_smiles_array: ndarray = MoleculeEncoder.encode(_data_df, _vocabulary, 90, molecule_encoding_type)

        # Assert that process is reversible
        for _i in range(_encoded_smiles_array.shape[0]):

            _encoded_smiles: str = _encoded_smiles_array[_i, :, :]

            # Decode features to SMILES
            _decoded_smiles: str = MoleculeEncoder.decode(
                _encoded_smiles,
                _inverse_vocabulary,
                molecule_encoding_type
            )

            assert _encoded_smiles == _decoded_smiles


if __name__ == "__main__":
    pytest.run()
