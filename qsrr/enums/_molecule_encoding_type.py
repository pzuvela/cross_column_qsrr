from enum import (
    Enum,
    unique
)


@unique
class MoleculeEncodingType(Enum):
    Integer = 1
    OneHot = 2
