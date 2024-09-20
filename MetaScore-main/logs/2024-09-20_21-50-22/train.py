import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
from tqdm import tqdm
import os
import shutil
import datetime

from Model.MetaScore import MetaScore
from Data.dataloader import create_data_loaders

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
        predictions = predictions.squeeze(-1)  # to remove the last dimension, output shape is [batch_size]
        loss = criterion(predictions, kd_values)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # record the loss of each batch
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
            predictions = predictions.squeeze(-1)  # guarantee the output shape is [batch_size]
            loss = criterion(predictions, kd_values)
            total_loss += loss.item()
            
            # record the loss of each batch
            logger.debug(f"Validation Batch Loss: {loss.item():.4f}")

    average_loss = total_loss / len(loader)
    return average_loss


def main():
    # get the current time as the log directory name
    current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_dir = os.path.join('logs', current_time)
    os.makedirs(log_dir, exist_ok=True)

    # configure the log recording
    log_file = os.path.join(log_dir, 'train.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger()

    # copy the model code and configuration file to the log directory
    # shutil.copytree('Model', os.path.join(log_dir, 'Model'))
    # shutil.copy('train.py', log_dir)
    # if you have a configuration file, you can copy it
    # shutil.copy('config.yml', log_dir)

    # other initialization code
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
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    # training epoch
    best_val_loss = float('inf')
    patience_counter = 0
    num_epochs = 100
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, logger)
        val_loss = validate(model, val_loader, criterion, device, logger)
        
        scheduler.step(val_loss)
        
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Epoch {epoch+1}/{num_epochs} | 
                    Train Loss: {train_loss:.4f} | 
                    Val Loss: {val_loss:.4f} | 
                    LR: {current_lr:.6f}")
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(log_dir, 'best_model.pth')
            torch.save(model.state_dict(), checkpoint_path)
            # logger.info(f"Saved best model to {checkpoint_path}")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # early stopping
        patience = 10
        if patience_counter >= patience:
            logger.info(f"Early stopping triggered after {epoch+1} epochs")
            break

    # evaluate the best model
    model.load_state_dict(torch.load(checkpoint_path))
    test_loss = validate(model, test_loader, criterion, device, logger)
    logger.info(f"Training completed. 
                Epoch {epoch+1}/{num_epochs} | 
                Train Loss: {train_loss:.4f} | 
                Val Loss: {val_loss:.4f} | 
                Test Loss: {test_loss:.4f}")

    print("Training completed.")

if __name__ == "__main__":
    main()