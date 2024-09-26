import torch
import torch.nn as nn
import torch.optim as optim

class LRScheduler:
    def __init__(self, optimizer, mode='min', factor=0.5, patience=10):
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=mode, factor=factor, patience=patience
        )

    def step(self, val_loss):
        self.scheduler.step(val_loss)

class LearnableLR(nn.Module):
    def __init__(self, model, init_lr=0.001):
        super(LearnableLR, self).__init__()
        """
        Initialize the learnable learning rate for each parameter.

        Args:
            model (nn.Module): The model to be optimized.
            init_lr (float): The initial learning rate.
        """
        self.lr_params = nn.ParameterDict()
        for name, param in model.named_parameters():
            lr = nn.Parameter(torch.full_like(param, init_lr))
            self.lr_params[name.replace('.', '_')] = lr

    def get_lrs(self):
        """
        Get the current learning rate dictionary.

        Returns:
            dict: The mapping from parameter names to learning rate tensors.
        """
        return {name.replace('_', '.'): lr for name, lr in self.lr_params.items()}
