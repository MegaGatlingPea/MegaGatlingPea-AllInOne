import torch.nn as nn

class KdPredictionModule(nn.Module):
    def __init__(self, input_dim):
        super(KdPredictionModule, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.mlp(x)