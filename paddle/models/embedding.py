import torch
import torch.nn as nn

class MoleculeEmbedding(nn.Module):
    def __init__(self, num_atom_features=9, num_bond_features=3, 
                 atom_embedding_dim=256, bond_embedding_dim=32, offset=128):
        super(MoleculeEmbedding, self).__init__()
        
        # use linear embedding to replace embedding layer
        self.atom_embedding = nn.Linear(num_atom_features, atom_embedding_dim)
        self.bond_embedding = nn.Linear(num_bond_features, bond_embedding_dim)

    def forward(self, x, edge_attr):
        # if input is integer index, need to convert to float
        x = x.float()
        edge_attr = edge_attr.float()
        
        x_embedded = self.atom_embedding(x)
        edge_attr_embedded = self.bond_embedding(edge_attr)
        
        return x_embedded, edge_attr_embedded