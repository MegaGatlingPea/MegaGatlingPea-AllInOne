# Model/Readout/mlp.py
import torch.nn as nn

class KdPredictionModule(nn.Module):
    def __init__(self, input_dim, hidden_dim1=32, hidden_dim2=16, batch_size=4):
        super(KdPredictionModule, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, 1)  # output dimension set to 1
        )

    def forward(self, x):
        out = self.mlp(x)
        return out.squeeze(-1)  # to remove the last dimension, output shape is [batch_size]