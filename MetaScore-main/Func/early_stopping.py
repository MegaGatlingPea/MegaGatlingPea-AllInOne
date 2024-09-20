import numpy as np
import torch

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', logger=None):
        """
        Args:
            patience (int): how many epochs to wait before stopping when loss is not improving.
            verbose (bool): whether to print message, default is False.
            delta (float): minimum change to consider as an improvement, default is 0.
            path (str): model save path, default is 'checkpoint.pt'.
            logger (Logger): logger instance.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.logger = logger

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.logger:
                self.logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """save model when validation loss decreases."""
        if self.verbose:
            msg = f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...'
            if self.logger:
                self.logger.info(msg)
            else:
                print(msg)
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
