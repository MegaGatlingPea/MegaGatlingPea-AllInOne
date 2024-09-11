import torch
from torch import nn

from GAT import GAT
from MPNN import MPNN
from Interaction import InteractionModule
from predict import KdPredictionModule

class MetaScore(nn.Module):
    def __init__(self, protein_input_dim, ligand_input_dim, protein_hidden_dim, ligand_hidden_dim, interaction_dim, ligand_edge_dim):
        super(MetaScore, self).__init__()
        self.protein_encoder = GAT(protein_input_dim, protein_hidden_dim, protein_hidden_dim)
        self.ligand_encoder = MPNN(ligand_input_dim, ligand_edge_dim, ligand_hidden_dim, ligand_hidden_dim)
        self.interaction_module = InteractionModule(protein_hidden_dim, ligand_hidden_dim, interaction_dim)
        self.kd_prediction = KdPredictionModule(interaction_dim)

    def forward(self, protein_data, ligand_data):
        protein_repr = self.protein_encoder(protein_data)
        ligand_repr = self.ligand_encoder(ligand_data)
        interaction_repr = self.interaction_module(protein_repr, ligand_repr)
        kd_prediction = self.kd_prediction(interaction_repr)
        return kd_prediction

# 使用示例
# model = MetaScore(
#     protein_input_dim=23,  # 蛋白质节点特征维度
#     ligand_input_dim=41,   # 配体节点特征维度
#     protein_hidden_dim=128,
#     ligand_hidden_dim=128,
#     interaction_dim=256,
#     ligand_edge_dim=11     # 配体边特征维度
# )

# # 在训练循环中
# for protein_batch, ligand_batch, kd_values, _ in dataloader:
#     predictions = model(protein_batch, ligand_batch)
#     loss = loss_function(predictions, kd_values)
#     # 反向传播和优化步骤...