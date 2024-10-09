# mlp.py

import torch
import torch.nn as nn

def get_activation(activation_name):
    activation_name = activation_name.lower()
    activations = {
        'relu': nn.ReLU(),
        'leakyrelu': nn.LeakyReLU(),
        'tanh': nn.Tanh(),
        'sigmoid': nn.Sigmoid(),
        'elu': nn.ELU(),
        'selu': nn.SELU(),
        'gelu': nn.GELU(),
        'relu6': nn.ReLU6(),        
    }

    if activation_name in activations:
        return activations[activation_name]
    else:
        raise ValueError(f"Unsupported activation function: {activation_name}")

# FCResNet
class FCResNet(nn.Module):
    def __init__(
        self,
        input_dim,
        features=256,
        depth=5,
        spectral_normalization=True,
        coeff=0.95,
        n_power_iterations=1,
        dropout_rate=0.05,
        num_outputs=1,
        activation="relu",
    ):
        super(FCResNet, self).__init__()
        self.input_layer = nn.Linear(input_dim, features)
        self.activation = get_activation(activation)
        self.dropout = nn.Dropout(dropout_rate)

        self.hidden_layers = nn.ModuleList()
        for _ in range(depth):
            layer = nn.Linear(features, features)
            if spectral_normalization:
                layer = nn.utils.spectral_norm(layer, n_power_iterations=n_power_iterations)
            self.hidden_layers.append(layer)

        self.output_layer = nn.Linear(features, num_outputs)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.activation(x)
        x = self.dropout(x)
        for layer in self.hidden_layers:
            residual = x
            x = layer(x)
            x = self.activation(x)
            x = self.dropout(x)
            x = x + residual  # Residual connection
        x = self.output_layer(x)
        return x

# Simple n-layer MLP
class SimpleMLP(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dims=[256, 256],
        num_outputs=1,
        activation="relu",
        dropout_rate=0.0,
        normalization=False
    ):
        super(SimpleMLP, self).__init__()
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            if normalization:
                layers.append(nn.BatchNorm1d(dim))
            layers.append(get_activation(activation))
            if dropout_rate > 0.0:
                layers.append(nn.Dropout(dropout_rate))
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, num_outputs))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# Simplest Linear Layer
class LinearModel(nn.Module):
    def __init__(self, input_dim, num_outputs=1):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(input_dim, num_outputs)

    def forward(self, x):
        return self.linear(x)

# ResNet (Alternative implementation)
class ResNet(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim=256,
        num_blocks=5,
        num_outputs=1,
        activation="relu",
        dropout_rate=0.0,
        normalization=False
    ):
        super(ResNet, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(ResBlock(hidden_dim, activation, dropout_rate, normalization))
        self.output_layer = nn.Linear(hidden_dim, num_outputs)
        self.activation = get_activation(activation)

    def forward(self, x):
        x = self.input_layer(x)
        for block in self.blocks:
            x = block(x)
        x = self.output_layer(x)
        return x

class ResBlock(nn.Module):
    def __init__(self, dim, activation="relu", dropout_rate=0.0, normalization=False):
        super(ResBlock, self).__init__()
        layers = []
        layers.append(nn.Linear(dim, dim))
        if normalization:
            layers.append(nn.BatchNorm1d(dim))
        layers.append(get_activation(activation))
        if dropout_rate > 0.0:
            layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.Linear(dim, dim))
        if normalization:
            layers.append(nn.BatchNorm1d(dim))
        self.block = nn.Sequential(*layers)
        self.activation = get_activation(activation)

    def forward(self, x):
        residual = x
        x = self.block(x)
        x += residual
        x = self.activation(x)
        return x