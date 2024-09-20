import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
from tqdm import tqdm

from Model.MetaScore import MetaScore
from Data.dataloader import create_data_loaders

def train_epoch(model, loader, optimizer, criterion, device):
    """to train an epoch"""
    model.train()
    total_loss = 0
    for protein_batch, ligand_batch, kd_values, _ in tqdm(loader, desc="Training"):
        protein_batch = protein_batch.to(device)
        ligand_batch = ligand_batch.to(device)
        kd_values = kd_values.to(device)

        optimizer.zero_grad()
        predictions = model(protein_batch, ligand_batch)
        loss = criterion(predictions, kd_values)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    return total_loss / len(loader)

def validate(model, loader, criterion, device):
    """to validate the model"""
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for protein_batch, ligand_batch, kd_values, _ in tqdm(loader, desc="Validating"):
            protein_batch = protein_batch.to(device)
            ligand_batch = ligand_batch.to(device)
            kd_values = kd_values.to(device)

            predictions = model(protein_batch, ligand_batch)
            loss = criterion(predictions, kd_values)
            total_loss += loss.item()
    
    return total_loss / len(loader)

def main():
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[logging.StreamHandler()]
    )
    logger = logging.getLogger()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)
        
        scheduler.step(val_loss)
        
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"{epoch+1} | {train_loss:.4f} | {val_loss:.4f} | {current_lr:.6f}")
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # early stopping
        patience = 10
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

    # evaluate the best model
    model.load_state_dict(torch.load("best_model.pth"))
    test_loss = validate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}")

    print("Training completed.")

if __name__ == "__main__":
    main()