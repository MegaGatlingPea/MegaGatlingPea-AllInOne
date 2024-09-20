import torch
from torch import nn
import torch.nn.functional as F

from Model.MoleculeEmbedding.embedding import MoleculeEmbedding
from Model.PocketEncoder.gat import GAT
from Model.MoleculeEncoder.mpnn import MPNN
from Model.InteractionBlock.interaction import InteractionModule
from Model.Readout.mlp import KdPredictionModule

class MetaScore(nn.Module):
    def __init__(self, num_atom_features=9, num_bond_features=3, 
                 atom_embedding_dim=256, bond_embedding_dim=64, 
                 protein_hidden_dim=512, ligand_hidden_dim=128, 
                 protein_output_dim=128, ligand_output_dim=128, 
                 interaction_dim=64):

        super(MetaScore, self).__init__()

        self.molecule_embedding = MoleculeEmbedding(
            num_atom_features, num_bond_features, 
            atom_embedding_dim, bond_embedding_dim)

        self.ligand_encoder = MPNN(input_dim=atom_embedding_dim, 
                                   edge_dim=bond_embedding_dim, 
                                   hidden_dim=ligand_hidden_dim,
                                   output_dim=ligand_output_dim,
                                   num_layers=3,
                                   activation=nn.ReLU(),
                                   dropout=0.1)
        
        self.protein_encoder = GAT(input_dim=1284, 
                                   hidden_dim=protein_hidden_dim,
                                   output_dim=protein_output_dim,
                                   num_layers=3,
                                   heads=4,
                                   activation=nn.ReLU(),
                                   dropout=0.2)
        
        self.interaction_module = InteractionModule(protein_dim=protein_output_dim, 
                                                    ligand_dim=ligand_output_dim, 
                                                    hidden_dim=interaction_dim)
        
        self.kd_prediction = KdPredictionModule(input_dim=interaction_dim,
                                                hidden_dim1=32,
                                                hidden_dim2=16)

    def forward(self, protein_data, ligand_data):
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