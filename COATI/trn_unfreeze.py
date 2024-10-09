# trn_unfreeze.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
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
import pickle  # For loading the serialized tokenizer
from coati.models.io.coati import load_e3gnn_smiles_clip_e2e
from coati.generative.coati_purifications import embed_smiles_batch

# Import MoleculePropertyDataset from func.py
from func import MoleculePropertyDataset4Smiles as MoleculePropertyDataset

# Import models from mlp.py
from mlp import FCResNet, SimpleMLP, ResNet, LinearModel

with open('./tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Set random seed for reproducibility
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

class CombinedModel(nn.Module):
    """Combined model integrating the encoder and predictor."""
    def __init__(self, encoder, predictor):
        super(CombinedModel, self).__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.tokenizer = tokenizer

    def forward(self, smiles):
        """
        Forward pass through the combined model.

        Args:
            smiles (list of str): List of SMILES strings.

        Returns:
            torch.Tensor: Predicted outputs.
        """
        encodings = embed_smiles_batch(smiles, self.encoder, self.tokenizer)  
        outputs = self.predictor(encodings)
        return outputs

class Trainer:
    """Trainer class for training the CombinedModel."""
    def __init__(self, config):
        self.tokenizer = tokenizer
        self.config = config
        set_seed(config['training_params']['seed'])
        self.DEVICE = torch.device(config['device'] if torch.cuda.is_available() else "cpu")
        self.current_time = datetime.now().strftime('%Y-%m-%d-%H-%M')
        # Create temporary model save directory
        self.temp_model_save_dir = os.path.join('pths', f"temp_{config['model_params']['model_name']}_{self.current_time}")
        os.makedirs(self.temp_model_save_dir, exist_ok=True)
        # Setup logging
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
        # Save used configuration
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

        # Initialize encoder, load tokenizer from serialized file, ensure encoder is trainable
        encoder, _ = load_e3gnn_smiles_clip_e2e(
            freeze=False,  # Ensure encoder is not frozen
            device=self.DEVICE,
            doc_url="./models/grande_closed.pkl",
        )
        # Unfreeze all encoder parameters
        for param in encoder.parameters():
            param.requires_grad = True

        # Initialize predictor based on configuration
        predictor = self._initialize_predictor(config, encoder)

        # Create combined model
        self.model = CombinedModel(encoder, predictor).to(self.DEVICE)

        # Initialize optimizer with all parameters of the combined model
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['training_params']['learning_rate']
        )

        self.criterion = nn.MSELoss()
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.2, 
            patience=config['training_params']['lr_scheduler_patience'], 
            verbose=True
        )

        # Early stopping parameters
        self.patience = config['training_params']['early_stopping_patience']
        self.early_stopping_delta = config['training_params']['early_stopping_delta']
        self.best_val_loss = float('inf')
        self.trigger_times = 0

        # Prepare dataset and dataloaders
        dataset = MoleculePropertyDataset(self.smiles, self.labels)
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

        self.model_save_dir = None

    def _initialize_predictor(self, config, encoder):
        """
        Initialize the predictor model based on configuration.

        Args:
            config (dict): Configuration dictionary.
            encoder (nn.Module): Encoder model.

        Returns:
            nn.Module: Initialized predictor model.
        """
        input_dim = 256  # Fixed embedding dimension
        model_name = config['model_params']['model_name']

        if model_name == 'FCResNet':
            predictor = FCResNet(
                input_dim=input_dim,
                features=config['model_params']['features'],
                depth=config['model_params']['depth'],
                spectral_normalization=config['model_params']['spectral_normalization'],
                coeff=config['model_params']['coeff'],
                n_power_iterations=config['model_params']['n_power_iterations'],
                dropout_rate=config['model_params']['dropout_rate'],
                num_outputs=config['model_params']['num_outputs'],
                activation=config['model_params']['activation']
            )
        elif model_name == 'SimpleMLP':
            predictor = SimpleMLP(
                input_dim=input_dim,
                hidden_dims=config['model_params'].get('hidden_dims', [256, 256]),
                num_outputs=config['model_params']['num_outputs'],
                activation=config['model_params']['activation'],
                dropout_rate=config['model_params']['dropout_rate'],
                normalization=config['model_params'].get('normalization', False)
            )
        elif model_name == 'ResNet':
            predictor = ResNet(
                input_dim=input_dim,
                hidden_dim=config['model_params']['features'],
                num_blocks=config['model_params']['depth'],
                num_outputs=config['model_params']['num_outputs'],
                activation=config['model_params']['activation'],
                dropout_rate=config['model_params']['dropout_rate'],
                normalization=config['model_params'].get('normalization', False)
            )
        elif model_name == 'LinearModel':
            predictor = LinearModel(
                input_dim=input_dim,
                num_outputs=config['model_params']['num_outputs']
            )
        else:
            raise ValueError(f"Unknown model name {model_name}")

        return predictor

    def train(self):
        """Training loop for the CombinedModel."""
        for epoch in range(self.config['training_params']['num_epochs']):
            self.model.train()
            train_loss = 0.0
            for batch in tqdm.tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config['training_params']['num_epochs']} - Training"):
                smiles_batch = batch['smiles']
                targets = batch['property'].to(self.DEVICE).unsqueeze(1)  # Ensure correct target shape

                self.optimizer.zero_grad()
                outputs = self.model(smiles_batch)  
                loss = self.criterion(outputs, targets)
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config['training_params']['grad_clip'])

                self.optimizer.step()

                train_loss += loss.item() * targets.size(0)
            
            train_loss /= len(self.train_loader.dataset)

            # Validation
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in tqdm.tqdm(self.val_loader, desc=f"Epoch {epoch+1}/{self.config['training_params']['num_epochs']} - Validation"):
                    smiles_batch = batch['smiles']
                    targets = batch['property'].to(self.DEVICE).unsqueeze(1)
                    outputs = self.model(smiles_batch)
                    loss = self.criterion(outputs, targets)
                    val_loss += loss.item() * targets.size(0)
            
            val_loss /= len(self.val_loader.dataset)

            self.scheduler.step(val_loss)

            # Get current learning rate
            lr = self.optimizer.param_groups[0]['lr']

            self.logger.info(f"Epoch {epoch+1}: LR = {lr:.6f}, Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

            # Check for early stopping and save the best model
            if val_loss < self.best_val_loss - self.early_stopping_delta:
                self.best_val_loss = val_loss
                best_model_path = os.path.join(self.temp_model_save_dir, 'best_model.pth')
                torch.save({
                    'model': self.model,  
                    'scaler': self.scaler
                }, best_model_path)
                self.trigger_times = 0
                self.logger.info(f"Best model saved with val_loss {self.best_val_loss:.4f}")
            else:
                self.trigger_times += 1
                if self.trigger_times >= self.patience:
                    self.logger.info("Early stopping triggered.")
                    break

        # Rename the model save directory after training
        time_str = datetime.now().strftime('%H-%M')
        new_folder_name = f"{self.config['model_params']['model_name']}_{self.best_val_loss:.4f}_{time_str}"
        new_model_save_dir = os.path.join('pths', new_folder_name)
        os.rename(self.temp_model_save_dir, new_model_save_dir)
        self.logger.info(f"Model directory renamed to {new_model_save_dir}")
        self.model_save_dir = new_model_save_dir

    def save_config(self):
        """Save the configuration file used for training."""
        config_save_path = os.path.join(self.temp_model_save_dir, 'config.yaml')
        source_config_path = './config.yaml'  # Assume original config file is at project root
        shutil.copy(source_config_path, config_save_path)
        self.logger.info(f"Config File has been saved to {config_save_path}")

class Inferencer:
    """Inferencer class for making predictions using the CombinedModel."""
    def __init__(self, config):
        self.config = config
        self.DEVICE = torch.device(config['device'] if torch.cuda.is_available() else "cpu")
        self.model_dir = config['inference_params']['model_dir']
        
        # load model and scaler
        model_path = os.path.join(self.model_dir, 'best_model.pth')
        checkpoint = torch.load(model_path, map_location=self.DEVICE)
        self.model = checkpoint['model'].to(self.DEVICE)
        self.model.eval()
        
        self.scaler = checkpoint['scaler']
        
        # Initialize tokenizer
        self.tokenizer = tokenizer  
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)  
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        if not self.logger.handlers:
            self.logger.addHandler(ch)

    def predict(self, smiles_list):
        """
        Make predictions on a list of SMILES strings.

        Args:
            smiles_list (list of str): List of SMILES strings.

        Returns:
            numpy.ndarray: Predicted property values.
        """
        self.model.to(self.DEVICE)
        outputs = self.model(smiles_list)
        outputs = outputs.cpu().detach().numpy()
        outputs_original = self.scaler.inverse_transform(outputs)
        self.model.to('cpu')
        torch.cuda.empty_cache()
        return outputs_original.flatten()
        
    def predict_from_csv(self, csv_path):
        """
        Make predictions from a CSV file containing SMILES strings.

        Args:
            csv_path (str): Path to the input CSV file.

        Returns:
            pd.DataFrame: DataFrame containing SMILES and their predicted properties.
        """
        try:
            df = pd.read_csv(csv_path)
            if 'smiles' not in df.columns:
                raise ValueError("CSV must contain 'smiles' column.")
            smiles_list = df['smiles'].tolist()

            self.logger.info("Begin Inference...")
            predictions = []
            for i in tqdm.tqdm(range(0, len(smiles_list), 64), desc="Predicting..."):
                batch_smiles = smiles_list[i:i+64]
                batch_predictions = self.predict(batch_smiles)
                predictions.extend(batch_predictions)
            
            df_ori = pd.read_csv('./paddle_prediction/data/test.csv')
            smiles_list_ori = df_ori['smiles'].tolist()
            
            results_df = pd.DataFrame({
                'smiles': smiles_list_ori,
                'pred': predictions
            })
            
            model_dir_last = os.path.basename(self.model_dir.rstrip('/'))
            output_csv = os.path.join(os.path.dirname(csv_path), f"{model_dir_last}.csv")
            
            results_df.to_csv(output_csv, index=False)
            self.logger.info(f"Inference result saves to {output_csv}")
            print(f"Inference result saves to {output_csv}")
            
            return results_df
        except Exception as e:
            self.logger.error(f"Error handling csv file: {e}")
            raise

if __name__ == "__main__":
    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    mode = config.get('mode', 'train')

    if mode == 'train':
        trainer = Trainer(config)
        trainer.train()
    elif mode == 'infer':
        inferencer = Inferencer(config)
        inference_params = config['inference_params']
        if 'input_csv' in inference_params:
            csv_path = inference_params['input_csv']
            results = inferencer.predict_from_csv(csv_path)
        else:
            smiles_list = inference_params['smiles_list']
            predictions = inferencer.predict(smiles_list)
            for smile, pred in zip(smiles_list, predictions):
                print(f"SMILES: {smile}, Predicted Property: {pred}")
    else:
        raise ValueError("Mode must be 'train' or 'infer'")