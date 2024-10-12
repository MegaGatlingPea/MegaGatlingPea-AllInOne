import torch
import torch.nn as nn
from models.embedding import MoleculeEmbedding
from models.mpnn import MPNN
from models.mlp import FCResNet, SimpleMLP, ResNet, LinearModel
import yaml
import os

def init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')  # 对于 ReLU
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.BatchNorm1d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
        
class MoleculeModel(nn.Module):
    def __init__(
        self,
        num_atom_features=9,
        num_bond_features=3,
        atom_embedding_dim=256,
        bond_embedding_dim=32,
        mpnn_hidden_dim=128,
        mpnn_output_dim=128,
        mpnn_num_layers=3,
        mlp_type='FCResNet',
        mlp_params=None,
        dropout=0.2
    ):
        """
        初始化MoleculeModel

        参数:
            num_atom_features (int): 原子特征维度
            num_bond_features (int): 键特征维度
            atom_embedding_dim (int): 原子嵌入维度
            bond_embedding_dim (int): 键嵌入维度
            mpnn_hidden_dim (int): MPNN隐藏层维度
            mpnn_output_dim (int): MPNN输出维度
            mpnn_num_layers (int): MPNN层数
            mlp_type (str): 选择的MLP类型 ('FCResNet', 'SimpleMLP', 'ResNet', 'LinearModel')
            mlp_params (dict): MLP的其他参数
            dropout (float): Dropout率
        """
        super(MoleculeModel, self).__init__()

        # 嵌入层
        self.embedding = MoleculeEmbedding(
            num_atom_features=num_atom_features,
            num_bond_features=num_bond_features,
            atom_embedding_dim=atom_embedding_dim,
            bond_embedding_dim=bond_embedding_dim
        )

        # MPNN层
        self.mpnn = MPNN(
            input_dim=atom_embedding_dim,
            edge_dim=bond_embedding_dim,
            hidden_dim=mpnn_hidden_dim,
            output_dim=mpnn_output_dim,
            num_layers=mpnn_num_layers,
            dropout=dropout
        )

        # 根据mlp_type选择对应的MLP模型
        if mlp_params is None:
            mlp_params = {}
        
        # 加载对应 mlp_type 的参数
        mlp_specific_params = mlp_params.get(mlp_type, {})
        mlp_specific_params['input_dim'] = mpnn_output_dim  # 确保 input_dim 一致

        if mlp_type == 'FCResNet':
            self.mlp = FCResNet(
                **mlp_specific_params
            )
        elif mlp_type == 'SimpleMLP':
            self.mlp = SimpleMLP(
                **mlp_specific_params
            )
        elif mlp_type == 'ResNet':
            self.mlp = ResNet(
                **mlp_specific_params
            )
        elif mlp_type == 'LinearModel':
            self.mlp = LinearModel(
                **mlp_specific_params
            )
        else:
            raise ValueError(f"Unsupported MLP type: {mlp_type}")

        self.apply(init_weights)
        
    def forward(self, data):
        """
        前向传播

        参数:
            data (Data): 包含图结构信息的PyG数据对象

        返回:
            torch.Tensor: 预测结果
        """
        x, edge_index, edge_attr, pos, batch = data.x, data.edge_index, data.edge_attr, data.pos, data.batch

        # 嵌入
        x_emb, edge_attr_emb = self.embedding(x, edge_attr)

        # MPNN
        mpnn_out = self.mpnn(
            data.__class__(x=x_emb, edge_index=edge_index, edge_attr=edge_attr_emb, pos=pos, batch=batch)
        )

        # MLP
        out = self.mlp(mpnn_out)

        return out
    
