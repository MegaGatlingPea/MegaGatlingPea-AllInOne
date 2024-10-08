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
