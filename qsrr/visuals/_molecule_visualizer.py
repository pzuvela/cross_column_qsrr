from rdkit import Chem
from rdkit.Chem import Draw

from matplotlib import colors


class Visualizer:

    @staticmethod
    def get_mol(smiles):
        """Returns SMILES String in RDKit molecule format"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        Chem.Kekulize(mol)
        return mol

    @staticmethod
    def get_image(mol, atomset, name):
        """Save image of the SMILES for vis purposes"""
        hcolor = colors.to_rgb('green')
        if atomset is not None:
            # highlight the atoms set while drawing the whole molecule.
            img = Draw.MolToImage(mol, size=(600, 600), fitImage=True, highlightAtoms=atomset, highlightColor=hcolor)
        else:
            img = Draw.MolToImage(mol, size=(400, 400), fitImage=True)

        img = img.save(name + ".jpg")
        return img
