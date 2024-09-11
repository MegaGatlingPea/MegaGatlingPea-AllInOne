import torch
import torch.nn as nn

class MoleculeEmbedding(nn.Module):
    def __init__(self, num_atom_features, num_bond_features, atom_embedding_dim, bond_embedding_dim, offset=128):
        super(MoleculeEmbedding, self).__init__()
        self.offset = offset
        
        # Atom embedding
        self.atom_embedding = nn.Embedding(num_atom_features * offset, atom_embedding_dim)
        
        # Bond embedding
        self.bond_embedding = nn.Embedding(num_bond_features * offset, bond_embedding_dim)

    def forward(self, x, edge_attr):
        # Embed atoms
        x_embedded = self.atom_embedding(x)
        
        # Embed bonds
        edge_attr_embedded = self.bond_embedding(edge_attr)
        
        return x_embedded, edge_attr_embedded