import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import datetime

from Model.MetaScore import MetaScore
from Data.dataloader import create_data_loaders
from Func.logger import Logger
from Func.early_stopping import EarlyStopping
from Func.lr_scheduler import LRScheduler

def train_epoch(model, loader, optimizer, criterion, device, logger):
    """to train an epoch"""
    model.train()
    total_loss = 0
    for protein_batch, ligand_batch, kd_values, _ in tqdm(loader, desc="Training"):
        optimizer.zero_grad()
        protein_batch = protein_batch.to(device)
        ligand_batch = ligand_batch.to(device)
        kd_values = kd_values.to(device)
        predictions = model(protein_batch, ligand_batch)
        predictions = predictions.squeeze(-1)
        loss = criterion(predictions, kd_values)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        logger.debug(f"Batch Loss: {loss.item():.4f}")

    average_loss = total_loss / len(loader)
    return average_loss

def validate(model, loader, criterion, device, logger):
    """to validate the model"""
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for protein_batch, ligand_batch, kd_values, _ in tqdm(loader, desc="Validating"):
            protein_batch = protein_batch.to(device)
            ligand_batch = ligand_batch.to(device)
            kd_values = kd_values.to(device)
            predictions = model(protein_batch, ligand_batch)
            predictions = predictions.squeeze(-1)
            loss = criterion(predictions, kd_values)
            total_loss += loss.item()
            
            logger.debug(f"Validation Batch Loss: {loss.item():.4f}")

    average_loss = total_loss / len(loader)
    return average_loss

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
        lmdb_path='./pdbbind.lmdb',
        batch_size=4,
        train_ratio=0.7,
        val_ratio=0.2,
        test_ratio=0.1,
        seed=42
    )

    # initialize the MetaScore model
    model = MetaScore(
        num_atom_features=9,
        num_bond_features=3,
        atom_embedding_dim=256,
        bond_embedding_dim=64,
        protein_hidden_dim=512,
        ligand_hidden_dim=128,
        protein_output_dim=128,
        ligand_output_dim=128,
        interaction_dim=64
    ).to(device)

    # loss and optimizer
    criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = LRScheduler(optimizer)

    # initialize EarlyStopping
    early_stopping = EarlyStopping(patience=10, verbose=True, logger=None, path=os.path.join(log_dir, 'best_model.pth'))

    # train loop
    num_epochs = 100
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, logger)
        val_loss = validate(model, val_loader, criterion, device, logger)

        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.6f}")

        # save the best model and check the early stopping condition
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            logger.info(f"Early stopping triggered after {epoch+1} epochs")
            break
        
        # Record the best model's epoch
        best_epoch = epoch + 1
        best_train_loss = train_loss
        best_val_loss = val_loss
        
    # load the best model and evaluate on the test set
    model.load_state_dict(torch.load(early_stopping.path))
    test_loss = validate(model, test_loader, criterion, device, logger)
    logger.info(f"Best Epoch: {best_epoch} | Train Loss: {best_train_loss:.4f} | Val Loss: {best_val_loss:.4f} | Test Loss: {test_loss:.4f}")

    logger.info("Training completed.")

if __name__ == "__main__":
    main()