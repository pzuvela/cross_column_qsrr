class Exceptions:

    @staticmethod
    def invalid_metric_type(*args, **kwargs):
        raise ValueError(
            "Invalid Metric Type !"
        )

    @staticmethod
    def invalid_molecule_encoding_type(*args, **kwargs):
        raise ValueError(
            "Invalid Molecule Encoding Type !"
        )
