import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.nn import LayerNorm

class GAT(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, heads=4, dropout=0.2):
        """
        Initialize the GAT.
        
        Args:
            input_dim (int): Dimension of input features
            hidden_dim (int): Dimension of hidden layers
            output_dim (int): Dimension of output features
            num_layers (int): Number of GAT layers
            heads (int): Number of attention heads in GAT layers
            dropout (float): Dropout rate
        """
        super(GAT, self).__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        self.gat_layers = nn.ModuleList()
        self.spatial_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.alphas = nn.ParameterList()
        
        for _ in range(num_layers):
            self.gat_layers.append(GATConv(hidden_dim, hidden_dim // heads, heads=heads, dropout=dropout))
            self.spatial_layers.append(nn.Linear(hidden_dim + 3, hidden_dim))
            self.layer_norms.append(LayerNorm(hidden_dim))
            self.alphas.append(nn.Parameter(torch.tensor(0.5)))
        
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, data):
        """
        Forward pass of the GAT.
        
        Args:
            data (Data): PyG Data object containing protein graph information
        
        Returns:
            torch.Tensor: Encoded protein representation
        """
        x, edge_index, pos, batch = data.x, data.edge_index, data.pos, data.batch
        
        # Initial projection of node features
        h = self.input_proj(x)
        
        for gat_layer, spatial_layer, layer_norm, alpha in zip(self.gat_layers, self.spatial_layers, self.layer_norms, self.alphas):
            # Graph attention update
            h_gat = gat_layer(h, edge_index)
            
            # Spatial information update
            spatial_info = torch.cat([h_gat, pos], dim=-1)
            h_spatial = spatial_layer(spatial_info)
            
            # Combine graph attention and spatial information with learnable weight
            h_new = alpha * h_gat + (1 - alpha) * h_spatial
            
            # Residual connection and layer normalization
            h = h + h_new
            h = layer_norm(h)
            h = F.elu(h)
        
        # Global pooling to get graph-level representation
        h_graph = global_mean_pool(h, batch)
        
        return self.output(h_graph)