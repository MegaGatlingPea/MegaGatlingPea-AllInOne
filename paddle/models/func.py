# func.py

import torch
from torch.utils.data import Dataset
import pickle
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
import numpy as np

class MoleculePropertyDataset(Dataset):
    def __init__(self, pkl_file, scaler=None):
        """
        初始化数据集

        参数:
            pkl_file (str): 包含数据的pickle文件路径
            scaler (StandardScaler, optional): 用于标签标准化的scaler
        """
        with open(pkl_file, 'rb') as f:
            self.data = pickle.load(f)
        self.keys = list(self.data.keys())
        self.scaler = scaler
        if self.scaler is not None:
            self.normalize_labels()

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        """
        获取指定索引的数据项

        参数:
            idx (int): 数据项的索引

        返回:
            torch_geometric.data.Data: 对应的图数据对象
        """
        data_item = self.data[self.keys[idx]]
        
        # 验证 data_item 是否为 Data 对象
        if not isinstance(data_item, Data):
            raise TypeError(f"Expected data_item to be torch_geometric.data.Data, but got {type(data_item)} at index {idx}")
        
        return data_item

    def normalize_labels(self):
        """
        使用 scaler 对标签进行标准化
        """
        all_labels = [data.y.item() for data in self]
        all_labels = np.array(all_labels).reshape(-1, 1)
        # 仅进行转换，不再拟合
        transformed_labels = self.scaler.transform(all_labels)
        for i, data in enumerate(self.data.values()):
            data.y = torch.tensor(transformed_labels[i], dtype=torch.float)

class MoleculePropertyDataset4Smiles(Dataset):
    def __init__(self, smiles_list, properties):
        """
        Parameters:
            smiles_list (list of str): List of SMILES strings.
            properties (torch.Tensor): Corresponding chemical property labels.
        """
        self.smiles = smiles_list
        self.properties = properties

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        return {
            'smiles': self.smiles[idx],
            'property': self.properties[idx]
        }
