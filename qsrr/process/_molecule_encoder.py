import collections
from typing import (
    Dict,
    List,
    Optional
)

from pandas import DataFrame
from numpy import ndarray
import numpy as np

from qsrr.common.constants import MoleculeEncoderConstants
from qsrr.enums import MoleculeEncodingType
from qsrr.exceptions import Exceptions


class MoleculeEncoder:

    @staticmethod
    def tokenize(smiles_string: str) -> List[str]:
        return [token for token in MoleculeEncoderConstants.SMI_REGEX.findall(smiles_string)]

    @staticmethod
    def load(file_path: str):

        """
        Loads a vocabulary file into a dictionary.
        """

        _vocabulary = collections.OrderedDict()

        with open(file_path, "r", encoding="utf-8") as f:
            _tokens = f.readlines()

        for _idx, _token in enumerate(_tokens):
            _token = _token.rstrip("\n")
            _vocabulary[_token] = _idx

        return _vocabulary

    @staticmethod
    def build(data: DataFrame, file_path: Optional[str] = None):

        if file_path is None:
            _vocabulary = set()
            _smiles_list = list(data['smiles'])
            for ex in _smiles_list:
                for letter in MoleculeEncoder.tokenize(ex):
                    _vocabulary.add(letter)
        else:
            _vocabulary = MoleculeEncoder.load(file_path)

        vocabulary = {
            '<PAD>': 0, 
            '<UNK>': 1
        }

        for _i, _letter in enumerate(_vocabulary):
            vocabulary[_letter] = _i + 2

        _inverse_vocabulary = {_val: _key for _key, _val in vocabulary.items()}
        _inverse_vocabulary[0] = ''

        return vocabulary, _inverse_vocabulary

    @staticmethod
    def encode(
        data: DataFrame,
        vocabulary: Dict[str, str],
        max_length: int = 90,
        encoding_type: MoleculeEncodingType = MoleculeEncodingType.Integer
    ) -> ndarray:

        """

        Encodes the smiles into a list of integers or one hot encoding

        """

        match encoding_type:

            case MoleculeEncodingType.Integer:

                _smiles_list = list(data['smiles'])

                for _i, _ex in enumerate(_smiles_list):

                    _smiles_list[_i] = MoleculeEncoder.tokenize(_ex)
                    _smiles_list[_i] = _smiles_list[_i][:max_length] + ['<PAD>'] * (max_length - len(_smiles_list[_i]))

                    _encoded_smiles_list = []

                    for _letter in _smiles_list[_i]:

                        if _letter in vocabulary:
                            _encoded_smiles_list.append(vocabulary[_letter])
                        else:
                            _encoded_smiles_list.append(vocabulary['<UNK>'])

                    _smiles_list[_i] = _encoded_smiles_list

                _smiles_arr = np.array(_smiles_list)

            case MoleculeEncodingType.OneHot:

                _smiles_arr = np.zeros((len(data), max_length, len(vocabulary)))

                for _i, _smiles in enumerate(data["smiles"]):

                    _smiles = MoleculeEncoder.tokenize(_smiles)
                    _smiles_length = len(_smiles)

                    _smiles = _smiles[:max_length] + ['<PAD>'] * (max_length - len(_smiles))

                    for _j, _letter in enumerate(_smiles):
                        if _letter in vocabulary:
                            _smiles_arr[_i, _j, vocabulary[_letter]] = 1
                        else:
                            _smiles_arr[_i, _j, vocabulary['<UNK>']] = 1

            case _:
                raise Exceptions.invalid_molecule_encoding_type()

        return _smiles_arr

    @staticmethod
    def decode(encoded: str, inverse_vocabulary: Dict[str, str], encoding_type: MoleculeEncodingType.OneHot) -> str:

        """

        Converts encoded feature to smiles

        """

        match encoding_type:
            case MoleculeEncodingType.OneHot:
                return "".join(inverse_vocabulary[let.item()] for let in encoded.argmax(axis=1))
            case MoleculeEncodingType.Integer:
                return "".join(inverse_vocabulary[let] for let in encoded)
            case _:
                raise Exceptions.invalid_molecule_encoding_type()
