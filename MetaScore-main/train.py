import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import gc
import datetime

from Model.MetaScore import MetaScore
from Data.dataloader import create_data_loaders
from Func.logger import Logger
from Func.early_stopping import EarlyStopping
from Func.lr_scheduler import LRScheduler
from Func.trainer import Trainer

import warnings
warnings.filterwarnings("ignore", message="TypedStorage is deprecated")

def main():
    # get the current time as the log directory name
    current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_dir = os.path.join('logs', current_time)
    os.makedirs(log_dir, exist_ok=True)

    # configure the logger
    logger = Logger(log_dir).get_logger()

    # initialize the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # create dataloader randomly
    train_loader, val_loader, test_loader = create_data_loaders(
        cluster_data_dir='./cluster_data',
        batch_size=4,
        train_ratio=0.7,
        val_ratio=0.2,
        test_ratio=0.1,
        seed=42,
        num_workers=4,
        pin_memory=True
    )

    # initialize the MetaScore model
    model = MetaScore(
        num_atom_features=9,
        num_bond_features=3,
        atom_embedding_dim=384,
        bond_embedding_dim=128,
        protein_hidden_dim=512,
        ligand_hidden_dim=512,
        protein_output_dim=256,
        ligand_output_dim=256,
        interaction_dim=128,
        interaction_hidden_dim1=64,
        interaction_hidden_dim2=64,
        dropout=0.2
    ).to(device)

    # loss and optimizer
    criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = LRScheduler(optimizer)

    # initialize Trainer
    trainer = Trainer(model, optimizer, criterion, device, logger, max_grad_norm=1.0, use_amp=False)

    # initialize EarlyStopping
    early_stopping = EarlyStopping(patience=60, verbose=True, logger=None, path=os.path.join(log_dir, 'best_model.pth'))

    # train loop
    num_epochs = 200
    best_epoch = 0
    best_train_loss = float('inf')
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        train_loss = trainer.train_epoch(train_loader)
        val_loss = trainer.validate(val_loader)

        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.6f}")

        # Clear cache and collect garbage
        torch.cuda.empty_cache()
        gc.collect()

        # Monitor memory usage
        # current_memory = torch.cuda.memory_allocated(device) / (1024 ** 3)
        # peak_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        # logger.info(f"Current Memory: {current_memory:.2f} GB | Peak Memory: {peak_memory:.2f} GB")

        # save the best model and check the early stopping condition
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            logger.info(f"Early stopping triggered after {epoch+1} epochs")
            break
        
        # Record the best model's epoch
        if val_loss < best_val_loss:
            best_epoch = epoch + 1
            best_train_loss = train_loss
            best_val_loss = val_loss
        
    # load the best model and evaluate on the test set
    model.load_state_dict(torch.load(early_stopping.path))
    test_loss = trainer.validate(test_loader)
    logger.info(f"Best Epoch: {best_epoch} | Train Loss: {best_train_loss:.4f} | Val Loss: {best_val_loss:.4f} | Test Loss: {test_loss:.4f}")

    logger.info("Training completed.")

if __name__ == "__main__":
    main()