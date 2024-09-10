import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.utils import add_self_loops

class MPNNLayer(MessagePassing):
    def __init__(self, hidden_dim, edge_dim, activation=F.relu, dropout=0.0):
        super(MPNNLayer, self).__init__(aggr='add')
        self.node_mlp = nn.Linear(hidden_dim, hidden_dim)
        self.message_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_dim + 3, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            activation(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.activation = activation
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
    def forward(self, x, edge_index, edge_attr, pos):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        self_loop_attr = torch.zeros((x.size(0), edge_attr.size(1)),
                                     device=edge_attr.device, dtype=edge_attr.dtype)
        edge_attr = torch.cat([edge_attr, self_loop_attr], dim=0)
        return self.propagate(edge_index, x=x, edge_attr=edge_attr, pos=pos)
    
    def message(self, x_i, x_j, edge_attr, pos_i, pos_j):
        rel_pos = pos_i - pos_j 
        return self.message_mlp(torch.cat([x_i, x_j, edge_attr, rel_pos], dim=1))
    
    def update(self, aggr_out, x):
        return self.dropout(self.activation(self.node_mlp(x) + aggr_out))

class MPNN(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, output_dim, num_layers=3, activation=F.relu, dropout=0.0):
        super(MPNN, self).__init__()
        self.input_proj = nn.Linear(node_dim, hidden_dim)
        
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(MPNNLayer(hidden_dim, edge_dim, activation, dropout))
        
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            activation(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, data):
        x, edge_index, edge_attr, pos, batch = data.x, data.edge_index, data.edge_attr, data.pos, data.batch
        
        h = self.input_proj(x)
        
        for layer in self.layers:
            h = h + layer(h, edge_index, edge_attr, pos)
        
        h = global_mean_pool(h, batch)
        return self.output(h)