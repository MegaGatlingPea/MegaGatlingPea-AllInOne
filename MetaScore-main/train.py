import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm

from utils.config import load_config
from MetaScore import MetaScore
from Dataset.Dataloader import create_data_loaders

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
    config = load_config('config.yaml')

    wandb.init(project=config.wandb.project, name=config.wandb.name, config=config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create dataloader randomly
    train_loader, val_loader, test_loader = create_data_loaders(
        config.data.lmdb_path, 
        config.data.batch_size
    )

    # initialize the MetaScore model
    model = MetaScore(
        protein_hidden_dim=config.model.protein_hidden_dim,
        ligand_hidden_dim=config.model.ligand_hidden_dim,
        interaction_dim=config.model.interaction_dim
    ).to(device)

    # loss and optimizer
    criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=config.training.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    # training epoch
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(config.training.num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)
        
        scheduler.step(val_loss)
        
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "learning_rate": optimizer.param_groups[0]['lr']
        })
        
        print(f"Epoch {epoch+1}/{config.training.num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # early stopping
        if patience_counter >= config.training.patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

    wandb.finish()

    # evaluate the best model
    model.load_state_dict(torch.load("best_model.pth"))
    test_loss = validate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}")

    print("Training completed.")

if __name__ == "__main__":
    main()