import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import gc

class Trainer:
    def __init__(self, model, optimizer, criterion, device, logger, max_grad_norm=1.0):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.logger = logger
        self.scaler = GradScaler()
        self.max_grad_norm = max_grad_norm

    def train_epoch(self, loader):
        """train an epoch"""
        self.model.train()
        total_loss = 0
        for protein_batch, ligand_batch, kd_values, _ in tqdm(loader, desc="Training"):
            self.optimizer.zero_grad()
            protein_batch = protein_batch.to(self.device, non_blocking=True)
            ligand_batch = ligand_batch.to(self.device, non_blocking=True)
            kd_values = kd_values.to(self.device, non_blocking=True)
            with autocast():
                predictions = self.model(protein_batch, ligand_batch)
                predictions = predictions.squeeze(-1)
                loss = self.criterion(predictions, kd_values)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            total_loss += loss.item()

            self.logger.debug(f"Batch Loss: {loss.item():.4f}")

            # clean up unnecessary variables
            del protein_batch, ligand_batch, kd_values, predictions, loss
            torch.cuda.empty_cache()
            gc.collect()

        average_loss = total_loss / len(loader)
        return average_loss

    def validate(self, loader):
        """validate the model"""
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for protein_batch, ligand_batch, kd_values, _ in tqdm(loader, desc="Validating"):
                protein_batch = protein_batch.to(self.device, non_blocking=True)
                ligand_batch = ligand_batch.to(self.device, non_blocking=True)
                kd_values = kd_values.to(self.device, non_blocking=True)
                predictions = self.model(protein_batch, ligand_batch)
                predictions = predictions.squeeze(-1)
                loss = self.criterion(predictions, kd_values)
                total_loss += loss.item()

                self.logger.debug(f"Validation Batch Loss: {loss.item():.4f}")

                # clean up unnecessary variables
                del protein_batch, ligand_batch, kd_values, predictions, loss
                torch.cuda.empty_cache()
                gc.collect()

        average_loss = total_loss / len(loader)
        return average_loss
