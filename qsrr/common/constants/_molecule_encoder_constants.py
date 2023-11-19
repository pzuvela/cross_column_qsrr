import re


class MoleculeEncoderConstants:

    SMI_REGEX = re.compile(
        r"""(\[[^]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|/|:|~|@|\?|>>?|\*|\$|%[0-9]{2}|[0-9])"""
    )
