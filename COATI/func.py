# func.py

import torch
from torch.utils.data import Dataset

class MoleculePropertyDataset(Dataset):
    def __init__(self, encodings, properties):
        """
        Parameters:
            encodings (torch.Tensor): Encoded vectors of molecules.
            properties (torch.Tensor): Corresponding chemical property labels.
        """
        self.encodings = encodings
        self.properties = properties

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        return {
            'encoding': self.encodings[idx],
            'property': self.properties[idx]
        }

class MoleculePropertyDataset4Smiles(Dataset):
    def __init__(self, smiles_list, properties):
        """
        Parameters:
            smiles_list (list of str): List of SMILES strings.
            properties (torch.Tensor): Corresponding chemical property labels.
        """
        self.smiles = smiles_list
        self.properties = properties

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        return {
            'smiles': self.smiles[idx],
            'property': self.properties[idx]
        }
