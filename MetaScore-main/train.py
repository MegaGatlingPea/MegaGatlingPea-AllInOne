import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm

from MetaScore import MetaScore
from Dataset.Dataloader import create_data_loaders

# Initialize wandb
wandb.init(project="MetaScore", name="experiment_1")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
BATCH_SIZE = 4
LEARNING_RATE = 1e-3
NUM_EPOCHS = 300
PATIENCE = 20  # Early stopping patience
LMDB_PATH = "path_to_your_lmdb_file"

# Create data loaders
train_loader, val_loader, test_loader = create_data_loaders(LMDB_PATH, BATCH_SIZE)

# Initialize model
model = MetaScore(
    protein_input_dim=23,  # Adjust according to your actual input dimensions
    ligand_input_dim=41,   
    protein_hidden_dim=128,
    ligand_hidden_dim=128,
    interaction_dim=256,
    ligand_edge_dim=11     
).to(device)

# Loss function and optimizer
criterion = nn.SmoothL1Loss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

# Training function
def train_epoch(model, loader, optimizer, criterion, device):
    """
    Train the model for one epoch.

    Args:
        model (nn.Module): The model to train
        loader (DataLoader): Training data loader
        optimizer (Optimizer): The optimizer
        criterion (nn.Module): The loss function
        device (torch.device): The device to use for computation

    Returns:
        float: Average loss for the epoch
    """
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

# Validation function
def validate(model, loader, criterion, device):
    """
    Validate the model.

    Args:
        model (nn.Module): The model to validate
        loader (DataLoader): Validation data loader
        criterion (nn.Module): The loss function
        device (torch.device): The device to use for computation

    Returns:
        float: Average loss for the validation set
    """
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

# Training loop
best_val_loss = float('inf')
patience_counter = 0

for epoch in range(NUM_EPOCHS):
    train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
    val_loss = validate(model, val_loader, criterion, device)
    
    scheduler.step(val_loss)
    
    wandb.log({
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "learning_rate": optimizer.param_groups[0]['lr']
    })
    
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_model.pth")
        patience_counter = 0
    else:
        patience_counter += 1
    
    # Early stopping
    if patience_counter >= PATIENCE:
        print(f"Early stopping triggered after {epoch+1} epochs")
        break

wandb.finish()

# Evaluate best model on test set
model.load_state_dict(torch.load("best_model.pth"))
test_loss = validate(model, test_loader, criterion, device)
print(f"Test Loss: {test_loss:.4f}")

print("Training completed.")