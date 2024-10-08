# main_training_script.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import tqdm
import pandas as pd
import yaml
import os
import logging
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import random
import numpy as np
import joblib
import shutil

# Import MoleculePropertyDataset from func.py
from func import MoleculePropertyDataset

# Import models from mlp.py
from mlp import FCResNet, SimpleMLP, ResNet, LinearModel

# Set random seed function
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Ensure reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class Trainer:
    def __init__(self, config):
        self.config = config
        set_seed(config['training_params']['seed'])
        self.DEVICE = torch.device(config['device'] if torch.cuda.is_available() else "cpu")
        self.current_time = datetime.now().strftime('%Y-%m-%d-%H-%M')
        # Create temporary model save directory
        self.temp_model_save_dir = os.path.join('pths', f"temp_{config['model_params']['model_name']}_{self.current_time}")
        if not os.path.exists(self.temp_model_save_dir):
            os.makedirs(self.temp_model_save_dir)
        # Set Logger
        log_filename = os.path.join(self.temp_model_save_dir, 'training.log')
        logging.basicConfig(
            level=logging.INFO,
            format='%(message)s',
            handlers=[
                logging.FileHandler(log_filename),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        # Save used config file
        self.save_config()

        # Load data
        smiles_data = pd.read_csv(config['train_csv'])
        self.smiles = list(smiles_data['smiles'])
        labels = smiles_data['label'].values.astype(float)
        # Normalize labels
        self.scaler = StandardScaler()
        labels = self.scaler.fit_transform(labels.reshape(-1, 1)).flatten()
        self.labels = torch.tensor(labels, dtype=torch.float32)

        # Save scaler for later use
        joblib.dump(self.scaler, os.path.join(self.temp_model_save_dir, 'label_scaler.save'))

        # Check if embeddings have been saved
        if os.path.exists('embeddings.pt'):
            self.logger.info("Loading embeddings from embeddings.pt")
            self.encodings = torch.load('embeddings.pt')
        else:
            # Load encoder and tokenizer, use CPU to save CUDA memory
            from coati.models.io.coati import load_e3gnn_smiles_clip_e2e
            from coati.generative.coati_purifications import embed_smiles_batch

            encoder, tokenizer = load_e3gnn_smiles_clip_e2e(
                freeze=True,
                device=torch.device('cpu'),
                doc_url="./models/grande_closed.pkl",
            )
            # Compute embeddings
            self.logger.info("Computing embeddings...")
            self.encodings = embed_smiles_batch(self.smiles, encoder, tokenizer)
            # Save embeddings to file
            torch.save(self.encodings, 'embeddings.pt')

        # Create dataset and data loader
        dataset = MoleculePropertyDataset(self.encodings, self.labels)
        train_size = int((1 - config['training_params']['validation_split']) * len(dataset))
        val_size = len(dataset) - train_size
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config['training_params']['batch_size'],
            shuffle=True,
            num_workers=config['training_params']['num_workers'],
            pin_memory=True
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=config['training_params']['batch_size'],
            shuffle=False,
            num_workers=config['training_params']['num_workers'],
            pin_memory=True
        )

        # Initialize model, loss function and optimizer
        input_dim = self.encodings.shape[1]

        # Choose model based on config
        model_name = config['model_params']['model_name']

        if model_name == 'FCResNet':
            self.model = FCResNet(
                input_dim=input_dim,
                features=config['model_params']['features'],
                depth=config['model_params']['depth'],
                spectral_normalization=config['model_params']['spectral_normalization'],
                coeff=config['model_params']['coeff'],
                n_power_iterations=config['model_params']['n_power_iterations'],
                dropout_rate=config['model_params']['dropout_rate'],
                num_outputs=config['model_params']['num_outputs'],
                activation=config['model_params']['activation']
            ).to(self.DEVICE)
        elif model_name == 'SimpleMLP':
            self.model = SimpleMLP(
                input_dim=input_dim,
                hidden_dims=config['model_params'].get('hidden_dims', [256, 256]),
                num_outputs=config['model_params']['num_outputs'],
                activation=config['model_params']['activation'],
                dropout_rate=config['model_params']['dropout_rate'],
                normalization=config['model_params'].get('normalization', False)
            ).to(self.DEVICE)
        elif model_name == 'ResNet':
            self.model = ResNet(
                input_dim=input_dim,
                hidden_dim=config['model_params']['features'],
                num_blocks=config['model_params']['depth'],
                num_outputs=config['model_params']['num_outputs'],
                activation=config['model_params']['activation'],
                dropout_rate=config['model_params']['dropout_rate'],
                normalization=config['model_params'].get('normalization', False)
            ).to(self.DEVICE)
        elif model_name == 'LinearModel':
            self.model = LinearModel(
                input_dim=input_dim,
                num_outputs=config['model_params']['num_outputs']
            ).to(self.DEVICE)
        else:
            raise ValueError(f"Unknown model name {model_name}")

        self.criterion = nn.MSELoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=config['training_params']['learning_rate'])

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.2, 
            patience=20, 
            verbose=True
        )

        # Training loop parameters
        self.num_epochs = config['training_params']['num_epochs']
        self.best_val_loss = float('inf')
        self.patience = config['training_params']['patience']
        self.trigger_times = 0
        self.grad_clip = config['training_params']['grad_clip']
        self.early_stopping_delta = config['training_params']['early_stopping_delta']

    def train(self):
        for epoch in range(self.num_epochs):
            self.model.train()
            train_loss = 0.0
            for batch in tqdm.tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs} - Training"):
                inputs = batch['encoding'].to(self.DEVICE)
                targets = batch['property'].to(self.DEVICE).unsqueeze(1)  # Ensure targets shape is correct

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)

                self.optimizer.step()

                train_loss += loss.item() * inputs.size(0)
            
            train_loss /= len(self.train_loader.dataset)

            # Validation
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in tqdm.tqdm(self.val_loader, desc=f"Epoch {epoch+1}/{self.num_epochs} - Validation"):
                    inputs = batch['encoding'].to(self.DEVICE)
                    targets = batch['property'].to(self.DEVICE).unsqueeze(1)  # Ensure targets shape is correct
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                    val_loss += loss.item() * inputs.size(0)
            
            val_loss /= len(self.val_loader.dataset)

            self.scheduler.step(val_loss)

            # Get current learning rate
            lr = self.optimizer.param_groups[0]['lr']

            self.logger.info(f"Epoch {epoch+1}: LR = {lr:.6f}, Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

            # Check for early stopping and save best model
            if val_loss < self.best_val_loss - self.early_stopping_delta:
                self.best_val_loss = val_loss
                best_model_path = os.path.join(self.temp_model_save_dir, 'best_model.pth')
                torch.save(self.model.state_dict(), best_model_path)
                self.trigger_times = 0
                self.logger.info(f"Best model saved with val_loss {self.best_val_loss:.4f}")
            else:
                self.trigger_times += 1
                if self.trigger_times >= self.patience:
                    self.logger.info("Early stopping triggered.")
                    break

        # After training, rename model save directory
        time_str = datetime.now().strftime('%H-%M')
        new_folder_name = f"{self.config['model_params']['model_name']}_{self.best_val_loss:.4f}_{time_str}"
        new_model_save_dir = os.path.join('pths', new_folder_name)
        os.rename(self.temp_model_save_dir, new_model_save_dir)
        self.logger.info(f"Model directory renamed to {new_model_save_dir}")
        # Update self.model_save_dir to new path
        self.model_save_dir = new_model_save_dir

    def save_config(self):
        # Save config file used for this training
        config_save_path = os.path.join(self.temp_model_save_dir, 'config.yaml')
        import shutil
        source_config_path = './config.yaml'  # Assuming original config file is in project root
        shutil.copy(source_config_path, config_save_path)
        self.logger.info(f"Config File has been saved to {config_save_path}")

class Inferencer:
    def __init__(self, config):
        self.config = config
        self.DEVICE = torch.device(config['device'] if torch.cuda.is_available() else "cpu")
        self.model_dir = config['inference_params']['model_dir']
        # Load scaler
        scaler_path = os.path.join(self.model_dir, 'label_scaler.save')
        self.scaler = joblib.load(scaler_path)
        # Load model
        model_path = os.path.join(self.model_dir, 'best_model.pth')
        # Get model parameters
        model_params_path = os.path.join(self.model_dir, 'config.yaml')
        with open(model_params_path, 'r') as f:
            model_config = yaml.safe_load(f)
        model_params = model_config['model_params']
        input_dim = config['inference_params'].get('input_dim')

        # Load encoder and tokenizer
        from coati.models.io.coati import load_e3gnn_smiles_clip_e2e
        from coati.generative.coati_purifications import embed_smiles_batch

        self.encoder, self.tokenizer = load_e3gnn_smiles_clip_e2e(
            freeze=True,
            device=torch.device('cpu'),
            doc_url="./models/grande_closed.pkl",
        )

        # If input_dim is not provided, compute an example embedding to get input_dim
        if input_dim is None:
            dummy_smiles = ['CC']  # Any valid SMILES string
            dummy_encodings = embed_smiles_batch(dummy_smiles, self.encoder, self.tokenizer)
            input_dim = dummy_encodings.shape[1]

        model_name = model_params['model_name']
        if model_name == 'FCResNet':
            self.model = FCResNet(
                input_dim=input_dim,
                features=model_params['features'],
                depth=model_params['depth'],
                spectral_normalization=model_params['spectral_normalization'],
                coeff=model_params['coeff'],
                n_power_iterations=model_params['n_power_iterations'],
                dropout_rate=model_params['dropout_rate'],
                num_outputs=model_params['num_outputs'],
                activation=model_params['activation']
            ).to(self.DEVICE)
        elif model_name == 'SimpleMLP':
            self.model = SimpleMLP(
                input_dim=input_dim,
                hidden_dims=model_params.get('hidden_dims', [256, 256]),
                num_outputs=model_params['num_outputs'],
                activation=model_params['activation'],
                dropout_rate=model_params['dropout_rate'],
                normalization=model_params.get('normalization', False)
            ).to(self.DEVICE)
        elif model_name == 'ResNet':
            self.model = ResNet(
                input_dim=input_dim,
                hidden_dim=model_params['features'],
                num_blocks=model_params['depth'],
                num_outputs=model_params['num_outputs'],
                activation=model_params['activation'],
                dropout_rate=model_params['dropout_rate'],
                normalization=model_params.get('normalization', False)
            ).to(self.DEVICE)
        elif model_name == 'LinearModel':
            self.model = LinearModel(
                input_dim=input_dim,
                num_outputs=model_params['num_outputs']
            ).to(self.DEVICE)
        else:
            raise ValueError(f"Unknown model name {model_name}")
        # Load model weights
        self.model.load_state_dict(torch.load(model_path, map_location=self.DEVICE))
        self.model.eval()

        self.encoder = self.encoder
        self.tokenizer = self.tokenizer

    def predict(self, smiles_list):
        # Compute embeddings
        from coati.generative.coati_purifications import embed_smiles_batch
        encodings = embed_smiles_batch(smiles_list, self.encoder, self.tokenizer)
        inputs = torch.tensor(encodings, dtype=torch.float32).to(self.DEVICE)
        with torch.no_grad():
            outputs = self.model(inputs)
        outputs = outputs.cpu().numpy()
        # Inverse normalize outputs
        outputs_original = self.scaler.inverse_transform(outputs)
        return outputs_original.flatten()

if __name__ == "__main__":
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    mode = config.get('mode', 'train')

    if mode == 'train':
        trainer = Trainer(config)
        trainer.train()
    elif mode == 'infer':
        inferencer = Inferencer(config)
        smiles_list = config['inference_params']['smiles_list']
        predictions = inferencer.predict(smiles_list)
        for smile, pred in zip(smiles_list, predictions):
            print(f"SMILES: {smile}, Predicted Property: {pred}")
    else:
        raise ValueError("Mode must be 'train' or 'infer'")
