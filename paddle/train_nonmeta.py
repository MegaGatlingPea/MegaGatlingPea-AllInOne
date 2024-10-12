import warnings

# Suppress specific UserWarnings
warnings.filterwarnings(
    "ignore",
    message="TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class.*",
    category=UserWarning
)

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
import tqdm
import yaml
import os
import logging
from datetime import datetime
import random
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Import custom modules
from models.func import MoleculePropertyDataset
from models.model import MoleculeModel

def set_seed(seed=42):
    """
    Set random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Ensure reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_logger(log_path, name=__name__):
    """
    Set up a logger.

    Parameters:
        log_path (str): Path to save the log file.
        name (str): Name of the logger.

    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')

    # Avoid adding duplicate handlers
    if not logger.handlers:
        fh = logging.FileHandler(log_path)
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)

        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)

        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger

class RMSELoss(nn.Module):
    """
    Root Mean Square Error Loss Function.
    """
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, yhat, y):
        return torch.sqrt(self.mse(yhat, y))

class Trainer:
    def __init__(self, config):
        self.config = config
        set_seed(config['training_params']['seed'])
        self.DEVICE = torch.device(config['device'] if torch.cuda.is_available() else "cpu")
        self.current_time = datetime.now().strftime('%Y-%m-%d-%H-%M')
        # Create model save directory
        self.model_save_dir = os.path.join('pths', f"{config['model_params']['mlp_type']}_{self.current_time}")
        os.makedirs(self.model_save_dir, exist_ok=True)
        # Set up logger
        self.logger = setup_logger(os.path.join(self.model_save_dir, 'training.log'), name='Trainer')
        # Save configuration
        self.save_config()

        # Load dataset
        dataset_path = config['dataset_path']
        self.logger.info(f"Loading dataset from {dataset_path}")
        dataset = MoleculePropertyDataset(dataset_path)

        # Split dataset into training and validation sets
        val_split = config['training_params']['validation_split']
        train_size = int((1 - val_split) * len(dataset))
        val_size = len(dataset) - train_size
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

        self.logger.info(f"Training set size: {len(self.train_dataset)}")
        self.logger.info(f"Validation set size: {len(self.val_dataset)}")

        # Read use_scaler configuration
        self.use_scaler = config['training_params'].get('use_scaler', True)

        # Initialize and apply scaler (if enabled)
        if self.use_scaler:
            self.scaler = StandardScaler()
            self.fit_scaler()
        else:
            self.scaler = None
            self.logger.info("Scaler is not used for label normalization.")

        # Create DataLoaders
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

        # Initialize model, loss function, and optimizer
        mlp_type = config['model_params']['mlp_type']
        mlp_params = config['model_params'].get('mlp_params', {})
        # Get parameters specific to mlp_type
        mlp_specific_params = mlp_params.get(mlp_type, {})

        self.model = MoleculeModel(
            num_atom_features=config['model_params'].get('num_atom_features', 9),
            num_bond_features=config['model_params'].get('num_bond_features', 3),
            atom_embedding_dim=config['model_params'].get('atom_embedding_dim', 256),
            bond_embedding_dim=config['model_params'].get('bond_embedding_dim', 32),
            mpnn_hidden_dim=config['model_params'].get('mpnn_hidden_dim', 128),
            mpnn_output_dim=config['model_params'].get('mpnn_output_dim', 128),
            mpnn_num_layers=config['model_params'].get('mpnn_num_layers', 3),
            mlp_type=mlp_type,
            mlp_params=mlp_params,
            dropout=config['model_params'].get('dropout', 0.2)
        ).to(self.DEVICE)

        self.criterion = RMSELoss()  # Use RMSELoss for loss calculation
        self.optimizer = optim.AdamW(self.model.parameters(), lr=config['training_params']['learning_rate'])

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.2,
            patience=10,
            verbose=True
        )

        # Training loop parameters
        self.num_epochs = config['training_params']['num_epochs']
        self.best_val_loss = float('inf')
        self.patience = config['training_params']['patience']
        self.trigger_times = 0
        self.grad_clip = config['training_params']['grad_clip']
        self.early_stopping_delta = config['training_params']['early_stopping_delta']

        # Loss log
        self.loss_log = {
            'train_loss': [],
            'val_loss': []
        }

    def save_config(self):
        # Save configuration file to save directory
        config_path = os.path.join(self.model_save_dir, 'config.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f)
        self.logger.info(f"Configuration file saved to {config_path}")

    def fit_scaler(self):
        if not self.use_scaler:
            return  # Skip if scaler is not used

        # Fit scaler using training labels only
        self.logger.info("Fitting StandardScaler using training labels only")
        all_train_labels = [data.y.item() for data in self.train_dataset]
        all_train_labels = np.array(all_train_labels).reshape(-1, 1)
        self.scaler.fit(all_train_labels)
        self.logger.info("Scaler fitted")

        # Apply scaler to training and validation datasets
        self.logger.info("Applying scaler for label normalization")
        self.train_dataset.dataset.scaler = self.scaler
        self.val_dataset.dataset.scaler = self.scaler
        self.train_dataset.dataset.normalize_labels()
        self.val_dataset.dataset.normalize_labels()

    def train(self):
        for epoch in range(self.num_epochs):
            train_loss = self.train_one_epoch()
            val_loss = self.validate()

            # Scheduler step
            self.scheduler.step(val_loss)

            # Current learning rate
            lr = self.optimizer.param_groups[0]['lr']

            self.logger.info(f"Epoch {epoch+1}: LR = {lr:.6f}, Train RMSE = {train_loss:.4f}, Val RMSE = {val_loss:.4f}")

            # Early stopping and save best model
            if val_loss < self.best_val_loss - self.early_stopping_delta:
                self.best_val_loss = val_loss
                best_model_path = os.path.join(self.model_save_dir, 'best_model.pth')
                torch.save(self.model.state_dict(), best_model_path)
                # Save scaler (if used)
                if self.use_scaler:
                    scaler_save_path = os.path.join(self.model_save_dir, 'scaler.pkl')
                    with open(scaler_save_path, 'wb') as f:
                        pickle.dump(self.scaler, f)
                    self.logger.info(f"Best model and scaler saved, val_rmse = {self.best_val_loss:.4f}")
                else:
                    self.logger.info(f"Best model saved, val_rmse = {self.best_val_loss:.4f}")
                self.trigger_times = 0
            else:
                self.trigger_times += 1
                self.logger.info(f"No improvement for {self.trigger_times} epochs")
                if self.trigger_times >= self.patience:
                    self.logger.info("Early stopping triggered.")
                    break

        # Save loss log
        loss_log_path = os.path.join(self.model_save_dir, 'loss_log.pkl')
        with open(loss_log_path, 'wb') as f:
            pickle.dump(self.loss_log, f)
        self.logger.info(f"Loss log saved to {loss_log_path}")

        # Generate and save loss curves
        self.plot_loss()

    def train_one_epoch(self):
        self.model.train()
        total_loss = 0.0
        for batch in tqdm.tqdm(self.train_loader, desc="Training"):
            data = batch.to(self.DEVICE)
            targets = data.y.unsqueeze(1)

            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.criterion(outputs, targets)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)

            self.optimizer.step()

            total_loss += loss.item() * data.num_graphs

        average_loss = total_loss / len(self.train_loader.dataset)
        self.loss_log['train_loss'].append(average_loss)
        return average_loss

    def validate(self):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in tqdm.tqdm(self.val_loader, desc="Validation"):
                data = batch.to(self.DEVICE)
                targets = data.y.unsqueeze(1)
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item() * data.num_graphs

        average_loss = total_loss / len(self.val_loader.dataset)
        self.loss_log['val_loss'].append(average_loss)
        return average_loss

    def plot_loss(self):
        epochs = range(1, len(self.loss_log['train_loss']) + 1)
        plt.figure(figsize=(8, 6))

        plt.plot(epochs, self.loss_log['train_loss'], label='Training RMSE')
        plt.plot(epochs, self.loss_log['val_loss'], label='Validation RMSE')
        plt.xlabel('Epoch')
        plt.ylabel('RMSE Loss')
        plt.title('Loss Curves')
        plt.legend()

        plt.tight_layout()
        loss_curve_path = os.path.join(self.model_save_dir, 'loss_curves.png')
        plt.savefig(loss_curve_path)
        plt.close()
        self.logger.info(f"Loss curves saved to {loss_curve_path}")

class Inferencer:
    def __init__(self, config):
        self.config = config
        self.DEVICE = torch.device(config['device'] if torch.cuda.is_available() else "cpu")
        self.model_save_dir = config['inference_params']['model_dir']
        self.logger = setup_logger(os.path.join(self.model_save_dir, 'inference.log'), name='Inferencer')
        self.load_config()
        self.load_model()
        self.load_scaler()

    def load_config(self):
        # Load configuration from model directory
        config_path = os.path.join(self.model_save_dir, 'config.yaml')
        with open(config_path, 'r') as f:
            self.model_config = yaml.safe_load(f)
        self.use_scaler = self.model_config['training_params'].get('use_scaler', True)

    def load_model(self):
        # Initialize model
        mlp_type = self.model_config['model_params']['mlp_type']
        mlp_params = self.model_config['model_params'].get('mlp_params', {})

        self.model = MoleculeModel(
            num_atom_features=self.model_config['model_params'].get('num_atom_features', 9),
            num_bond_features=self.model_config['model_params'].get('num_bond_features', 3),
            atom_embedding_dim=self.model_config['model_params'].get('atom_embedding_dim', 256),
            bond_embedding_dim=self.model_config['model_params'].get('bond_embedding_dim', 32),
            mpnn_hidden_dim=self.model_config['model_params'].get('mpnn_hidden_dim', 128),
            mpnn_output_dim=self.model_config['model_params'].get('mpnn_output_dim', 128),
            mpnn_num_layers=self.model_config['model_params'].get('mpnn_num_layers', 3),
            mlp_type=mlp_type,
            mlp_params=mlp_params,
            dropout=self.model_config['model_params'].get('dropout', 0.2)
        ).to(self.DEVICE)

        # Load best model weights
        best_model_path = os.path.join(self.model_save_dir, 'best_model.pth')
        self.model.load_state_dict(torch.load(best_model_path, map_location=self.DEVICE))
        self.model.eval()
        self.logger.info(f"Model loaded from {best_model_path}")

    def load_scaler(self):
        if self.use_scaler:
            scaler_path = os.path.join(self.model_save_dir, 'scaler.pkl')
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                self.logger.info(f"Scaler loaded from {scaler_path}")
            else:
                self.scaler = None
                self.logger.warning(f"Scaler file {scaler_path} does not exist. Predictions will not be unstandardized.")
        else:
            self.scaler = None
            self.logger.info("Scaler was not used during training. Skipping unstandardization.")

    def predict_from_pkl(self, pkl_path):
        """
        从pkl文件中加载数据并进行预测。

        参数:
            pkl_path (str): pkl文件的路径，包含一个字典，键是索引，值是图数据。

        返回:
            pd.DataFrame: 包含SMILES和预测值的DataFrame。
        """
        try:
            with open(pkl_path, 'rb') as f:
                data_dict = pickle.load(f)
            self.logger.info(f"从 {pkl_path} 加载数据成功，共有 {len(data_dict)} 个样本。")
            
            data_list = list(data_dict.values())
            if not data_list:
                self.logger.error("pkl文件中没有有效的数据。")
                return None

            # 创建DataLoader
            batch_size = self.config['training_params']['batch_size']
            data_loader = DataLoader(data_list, batch_size=batch_size, shuffle=False)
            predictions = []
            indices = list(data_dict.keys())

            self.logger.info("开始进行预测...")
            for batch in tqdm.tqdm(data_loader, desc="预测中"):
                batch = batch.to(self.DEVICE)
                with torch.no_grad():
                    outputs = self.model(batch)
                outputs = outputs.cpu().numpy()
                # 逆归一化
                if self.use_scaler and self.scaler:
                    outputs_original = self.scaler.inverse_transform(outputs)
                else:
                    outputs_original = outputs
                predictions.extend(outputs_original.flatten())

            # 准备结果DataFrame
            import pandas as pd
            df_ori = pd.read_csv('/home/megagatlingpea/workdir/MegaGatlingPea-AllInOne/paddle/data/raw/test.csv')
            smiles = df_ori['smiles'].tolist()
            results_df = pd.DataFrame({
                'smiles': smiles,
                'pred': predictions
            })

            # 确定输出CSV路径
            model_dir_last = os.path.basename(os.path.normpath(self.model_save_dir))
            output_csv = os.path.join(os.path.dirname(pkl_path), f"{model_dir_last}_predictions.csv")

            # 保存结果
            results_df.to_csv(output_csv, index=False)
            self.logger.info(f"预测结果已保存到 {output_csv}")
            print(f"预测结果已保存到 {output_csv}")

            return results_df
        except Exception as e:
            self.logger.error(f"处理pkl文件时出错: {e}")
            raise

    def predict(self, batch):
        with torch.no_grad():
            outputs = self.model(batch)
        outputs = outputs.cpu().numpy()
        if self.use_scaler and self.scaler:
            outputs_original = self.scaler.inverse_transform(outputs)
        else:
            outputs_original = outputs
        return outputs_original.flatten()

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
        if 'input_pkl' in inference_params:
            pkl_path = inference_params['input_pkl']
            results = inferencer.predict_from_pkl(pkl_path)
        else:
            # If no pkl input for prediction, define other prediction methods as needed
            raise NotImplementedError("Inference mode requires 'input_pkl' in configuration.")
    else:
        raise ValueError("Mode must be 'train' or 'infer'")
    