import torch
from torch import nn

from ProteinEncoder.GAT import GAT
from MoleculeEncoder.MPNN import MPNN
from InteractionBlock.Interaction import InteractionModule
from predict import KdPredictionModule

class MoleculeEmbedding(nn.Module):
    def __init__(self, num_atom_features, num_bond_features, embedding_dim, offset=128):
        super(MoleculeEmbedding, self).__init__()
        self.offset = offset
        
        # Atom embedding
        self.atom_embedding = nn.Embedding(num_atom_features * offset, embedding_dim)
        
        # Bond embedding
        self.bond_embedding = nn.Embedding(num_bond_features * offset, embedding_dim)

    def forward(self, x, edge_attr):
        # Embed atoms
        x_embedded = self.atom_embedding(x)
        
        # Embed bonds
        edge_attr_embedded = self.bond_embedding(edge_attr)
        
        return x_embedded, edge_attr_embedded

class MetaScore(nn.Module):
    def __init__(self, protein_hidden_dim, ligand_hidden_dim, interaction_dim):
        super(MetaScore, self).__init__()
        self.protein_encoder = GAT(protein_hidden_dim, protein_hidden_dim)
        
        self.molecule_embedding = None
        self.ligand_encoder = MPNN(ligand_hidden_dim, ligand_hidden_dim, ligand_hidden_dim)
        
        self.interaction_module = InteractionModule(protein_hidden_dim, ligand_hidden_dim, interaction_dim)
        self.kd_prediction = KdPredictionModule(interaction_dim)

    def forward(self, protein_data, ligand_data):
        # Initialize molecule embedding if not already done
        if self.molecule_embedding is None:
            num_atom_features = ligand_data.x.size(1)
            num_bond_features = ligand_data.edge_attr.size(1)
            self.molecule_embedding = MoleculeEmbedding(num_atom_features, num_bond_features, self.ligand_encoder.input_dim)

        # Encode protein
        protein_repr = self.protein_encoder(protein_data)
        
        # Embed and encode ligand
        x_embedded, edge_attr_embedded = self.molecule_embedding(ligand_data.x, ligand_data.edge_attr)
        ligand_data.x = x_embedded
        ligand_data.edge_attr = edge_attr_embedded
        ligand_repr = self.ligand_encoder(ligand_data)
        
        # Interaction and prediction
        interaction_repr = self.interaction_module(protein_repr, ligand_repr)
        kd_prediction = self.kd_prediction(interaction_repr)
        return kd_prediction