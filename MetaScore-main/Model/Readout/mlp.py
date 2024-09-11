import torch.nn as nn

class KdPredictionModule(nn.Module):
    def __init__(self, input_dim, hidden_dim1=32, hidden_dim2=16):
        super(KdPredictionModule, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, 1)
        )

    def forward(self, x):
        return self.mlp(x)