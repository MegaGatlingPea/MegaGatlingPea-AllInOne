import random
import lmdb
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch

class PDBBindDataset(Dataset):
    def __init__(self, lmdb_path, pdb_ids=None):
        """
        Initialize the PDBBindDataset.

        Args:
            lmdb_path (str): Path to the LMDB file.
            pdb_ids (list): List of PDB IDs to use. If None, use all PDB IDs in the database.
        """
        self.env = lmdb.open(lmdb_path, readonly=True, lock=False)
        self.txn = self.env.begin()
        if pdb_ids is None:
            self.pdb_ids = self._load_pdb_ids()
        else:
            self.pdb_ids = pdb_ids

    def _load_pdb_ids(self):
        """Load all PDB IDs from the LMDB file."""
        pdb_ids = []
        for key, value in self.txn.cursor():
            cluster_data = pickle.loads(value)
            pdb_ids.extend(list(cluster_data.keys()))
        return pdb_ids

    def __len__(self):
        """Return the number of items in the dataset."""
        return len(self.pdb_ids)

    def __getitem__(self, idx):
        """
        Get an item from the dataset.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            tuple: (protein_graph, ligand_graph, kd_value, pdb_id)
        """
        pdb_id = self.pdb_ids[idx]
        for key, value in self.txn.cursor():
            cluster_data = pickle.loads(value)
            if pdb_id in cluster_data:
                item = cluster_data[pdb_id]
                protein_graph = item['protein_graph']
                ligand_graph = item['ligand_graph']
                kd_value = torch.tensor([item['kd_value']], dtype=torch.float)
                return protein_graph, ligand_graph, kd_value, pdb_id
        raise ValueError(f"PDB ID {pdb_id} not found in the database")

    def __del__(self):
        """Close the LMDB environment when the object is deleted."""
        self.env.close()

def create_data_loaders(lmdb_path, batch_size, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, seed=42):
    """
    Create DataLoaders for train, validation, and test sets.

    Args:
        lmdb_path (str): Path to the LMDB file.
        batch_size (int): Batch size for the DataLoaders.
        train_ratio (float): Ratio of data to use for training.
        val_ratio (float): Ratio of data to use for validation.
        test_ratio (float): Ratio of data to use for testing.
        seed (int): Random seed for reproducibility.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Create full dataset
    full_dataset = PDBBindDataset(lmdb_path)
    
    # Set random seed for reproducibility
    random.seed(seed)
    torch.manual_seed(seed)
    
    # Get all PDB IDs and shuffle them
    all_pdb_ids = full_dataset.pdb_ids
    random.shuffle(all_pdb_ids)
    
    # Calculate sizes for each set
    dataset_size = len(all_pdb_ids)
    train_size = int(train_ratio * dataset_size)
    val_size = int(val_ratio * dataset_size)
    test_size = dataset_size - train_size - val_size
    
    # Split PDB IDs
    train_pdb_ids = all_pdb_ids[:train_size]
    val_pdb_ids = all_pdb_ids[train_size:train_size+val_size]
    test_pdb_ids = all_pdb_ids[train_size+val_size:]
    
    # Create subdatasets
    train_dataset = PDBBindDataset(lmdb_path, train_pdb_ids)
    val_dataset = PDBBindDataset(lmdb_path, val_pdb_ids)
    test_dataset = PDBBindDataset(lmdb_path, test_pdb_ids)
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    return train_loader, val_loader, test_loader

def collate_fn(batch):
    """
    Custom collate function for batching graph data.

    Args:
        batch (list): List of tuples (protein_graph, ligand_graph, kd_value, pdb_id)

    Returns:
        tuple: (batched_protein_graphs, batched_ligand_graphs, batched_kd_values, pdb_ids)
    """
    protein_graphs, ligand_graphs, kd_values, pdb_ids = zip(*batch)
    
    batched_protein_graphs = Batch.from_data_list(protein_graphs)
    batched_ligand_graphs = Batch.from_data_list(ligand_graphs)
    batched_kd_values = torch.cat(kd_values)
    
    return batched_protein_graphs, batched_ligand_graphs, batched_kd_values, pdb_ids

# Usage example
if __name__ == "__main__":
    LMDB_PATH = "path_to_your_lmdb_file"
    BATCH_SIZE = 32

    train_loader, val_loader, test_loader = create_data_loaders(LMDB_PATH, BATCH_SIZE)

    print(f"Number of batches in train_loader: {len(train_loader)}")
    print(f"Number of batches in val_loader: {len(val_loader)}")
    print(f"Number of batches in test_loader: {len(test_loader)}")

    # Example of iterating through the train_loader
    for protein_batch, ligand_batch, kd_values, pdb_ids in train_loader:
        print(f"Protein batch shape: {protein_batch.x.shape}")
        print(f"Ligand batch shape: {ligand_batch.x.shape}")
        print(f"Kd values shape: {kd_values.shape}")
        print(f"Number of PDB IDs in batch: {len(pdb_ids)}")
        break  # Just print the first batch and break