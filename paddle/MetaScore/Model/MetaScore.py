import torch
from torch import nn
import torch.nn.functional as F
import copy

from Model.MoleculeEmbedding.embedding import MoleculeEmbedding
from Model.PocketEncoder.gat import GAT
from Model.MoleculeEncoder.mpnn import MPNN
from Model.InteractionBlock.interaction import InteractionModule
from Model.Readout.mlp import KdPredictionModule

class MetaScore(nn.Module):
    def __init__(self, num_atom_features=9, num_bond_features=3, 
                 atom_embedding_dim=512, bond_embedding_dim=128, 
                 protein_hidden_dim=512, ligand_hidden_dim=512, 
                 protein_output_dim=256, ligand_output_dim=256, 
                 interaction_dim=128, interaction_hidden_dim1=64, interaction_hidden_dim2=64,
                 dropout=0.2):

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
                                   dropout=dropout)
        
        self.protein_encoder = GAT(input_dim=1280, 
                                   hidden_dim=protein_hidden_dim,
                                   output_dim=protein_output_dim,
                                   num_layers=3,
                                   heads=4,
                                   activation=nn.ReLU(),
                                   dropout=dropout)
        
        self.interaction_module = InteractionModule(protein_dim=protein_output_dim, 
                                                    ligand_dim=ligand_output_dim, 
                                                    hidden_dim=interaction_dim)
        
        self.kd_prediction = KdPredictionModule(input_dim=interaction_dim,
                                                hidden_dim1=interaction_hidden_dim1,
                                                hidden_dim2=interaction_hidden_dim2)

    def forward(self, protein_data, ligand_data):
        # Encode protein
        protein_repr = self.protein_encoder(protein_data)
        
        # Embed and encode ligand
        x_embedded, edge_attr_embedded = self.molecule_embedding(ligand_data.x, ligand_data.edge_attr)
        
        # clone ligand_data, avoid in-place modification
        ligand_data_embedded = copy.deepcopy(ligand_data)
        ligand_data_embedded.x = x_embedded
        ligand_data_embedded.edge_attr = edge_attr_embedded
        
        # use cloned data for encoding
        ligand_repr = self.ligand_encoder(ligand_data_embedded)
        
        # Interaction and prediction
        interaction_repr = self.interaction_module(protein_repr, ligand_repr)
        kd_prediction = self.kd_prediction(interaction_repr)
        return kd_prediction