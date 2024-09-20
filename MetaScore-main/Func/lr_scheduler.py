import torch.optim as optim

class LRScheduler:
    def __init__(self, optimizer, mode='min', factor=0.5, patience=10):
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=mode, factor=factor, patience=patience
        )

    def step(self, val_loss):
        self.scheduler.step(val_loss)
