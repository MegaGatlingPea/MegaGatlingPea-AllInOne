import os
import random
import lmdb
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch
import logging

class PDBBindDataset(Dataset):
    def __init__(self, cluster_data_dir, pdb_ids=None):
        """
        Initialize PDBBindDataset。

        Args:
            cluster_data_dir (str): file paths of all cluster LMDB files.
            pdb_ids (list, optional): list of PDB IDs to use. If None, all PDB IDs in all LMDB files will be used.
        """
        self.cluster_data_dir = cluster_data_dir
        self.lmdb_paths = self._get_all_lmdb_paths()

        # map PDB IDs to their corresponding LMDB paths
        self.pdb_id_to_lmdb_path = self._map_pdb_ids_to_lmdb_path()

        if pdb_ids is not None:
            # if specific PDB ID list is provided, filter it
            self.pdb_ids = [pdb_id for pdb_id in pdb_ids if pdb_id in self.pdb_id_to_lmdb_path]
        else:
            # use all available PDB IDs
            self.pdb_ids = list(self.pdb_id_to_lmdb_path.keys())

    def _get_all_lmdb_paths(self):
        """get file paths of all LMDB files under cluster_data_dir."""
        lmdb_files = []
        for fname in os.listdir(self.cluster_data_dir):
            path = os.path.join(self.cluster_data_dir, fname)
            if os.path.isdir(path) and fname.endswith('.lmdb'):
                lmdb_files.append(path)
            elif os.path.isfile(path) and fname.endswith('.lmdb'):
                lmdb_files.append(path)
        return lmdb_files

    def _map_pdb_ids_to_lmdb_path(self):
        """create a mapping of PDB IDs to their corresponding LMDB file paths."""
        pdb_id_to_lmdb_path = {}
        for lmdb_path in self.lmdb_paths:
            env = lmdb.open(lmdb_path, readonly=True, lock=False)
            with env.begin() as txn:
                cursor = txn.cursor()
                for key, _ in cursor:
                    pdb_id = key.decode()
                    pdb_id_to_lmdb_path[pdb_id] = lmdb_path
            env.close()
        return pdb_id_to_lmdb_path

    def __len__(self):
        """return the number of samples in the dataset."""
        return len(self.pdb_ids)

    def __getitem__(self, idx):
        """
        get a sample from the dataset.

        Args:
            idx (int): index of the sample to retrieve.

        Returns:
            tuple: (protein_graph, ligand_graph, kd_value, pdb_id)
        """
        pdb_id = self.pdb_ids[idx]
        lmdb_path = self.pdb_id_to_lmdb_path[pdb_id]

        # open the corresponding LMDB environment
        env = lmdb.open(lmdb_path, readonly=True, lock=False)
        with env.begin() as txn:
            data = txn.get(pdb_id.encode())
            if data is not None:
                item = pickle.loads(data)
            else:
                env.close()
                raise KeyError(f"PDB ID {pdb_id} not found in LMDB.")
        env.close()

        protein_graph = item['protein_graph']
        ligand_graph = item['ligand_graph']
        kd_value = torch.tensor([item['kd_value']], dtype=torch.float)
        return protein_graph, ligand_graph, kd_value, pdb_id

def create_data_loaders(cluster_data_dir, batch_size=4, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, seed=42, num_workers=4, pin_memory=True):
    """
    create data loaders for training, validation, and testing.

    Args:
        cluster_data_dir (str): directory path containing all cluster LMDB files.
        batch_size (int): batch size of the data loader.
        train_ratio (float): proportion of the training set.
        val_ratio (float): proportion of the validation set.
        test_ratio (float): proportion of the test set.
        seed (int): random seed to ensure reproducibility.
        num_workers (int): number of subprocesses for data loading.
        pin_memory (bool): whether to pin memory for faster data transfer.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # set random seed to ensure reproducibility
    random.seed(seed)
    torch.manual_seed(seed)

    # create full dataset
    full_dataset = PDBBindDataset(cluster_data_dir)

    # get all PDB IDs and shuffle
    all_pdb_ids = full_dataset.pdb_ids
    random.shuffle(all_pdb_ids)

    # calculate the size of training, validation, and test sets
    dataset_size = len(all_pdb_ids)
    train_size = int(train_ratio * dataset_size)
    val_size = int(val_ratio * dataset_size)
    test_size = dataset_size - train_size - val_size

    # split PDB IDs
    train_pdb_ids = all_pdb_ids[:train_size]
    val_pdb_ids = all_pdb_ids[train_size:train_size + val_size]
    test_pdb_ids = all_pdb_ids[train_size + val_size:]

    # create sub datasets
    train_dataset = PDBBindDataset(cluster_data_dir, train_pdb_ids)
    val_dataset = PDBBindDataset(cluster_data_dir, val_pdb_ids)
    test_dataset = PDBBindDataset(cluster_data_dir, test_pdb_ids)

    # create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             collate_fn=collate_fn, num_workers=num_workers, pin_memory=pin_memory)

    # add logging (optional)
    logger = logging.getLogger()
    logger.info(f"Training set size: {len(train_dataset)}")
    logger.info(f"Validation set size: {len(val_dataset)}")
    logger.info(f"Test set size: {len(test_dataset)}")

    return train_loader, val_loader, test_loader

def collate_fn(batch):
    """
    custom collate function for batching graph data.

    Args:
        batch (list): list of tuples (protein_graph, ligand_graph, kd_value, pdb_id).

    Returns:
        tuple: (batched_protein_graphs, batched_ligand_graphs, batched_kd_values, pdb_ids)
    """
    protein_graphs, ligand_graphs, kd_values, pdb_ids = zip(*batch)

    batched_protein_graphs = Batch.from_data_list(protein_graphs)
    batched_ligand_graphs = Batch.from_data_list(ligand_graphs)
    batched_kd_values = torch.cat(kd_values)

    return batched_protein_graphs, batched_ligand_graphs, batched_kd_values, pdb_ids

# example usage
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    CLUSTER_DATA_DIR = "./../cluster_data"  # update this path to your actual data directory
    BATCH_SIZE = 4

    train_loader, val_loader, test_loader = create_data_loaders(CLUSTER_DATA_DIR, BATCH_SIZE)

    print(f"Training loader batch size: {len(train_loader)}")
    print(f"Validation loader batch size: {len(val_loader)}")
    print(f"Test loader batch size: {len(test_loader)}")

    # iterate over the first batch of train_loader
    for protein_batch, ligand_batch, kd_values, pdb_ids in train_loader:
        print(f"Protein batch shape: {protein_batch.x.shape}")
        print(f"Ligand batch shape: {ligand_batch.x.shape}")
        print(f"Kd values shape: {kd_values.shape}")
        print(f"Number of PDB IDs in the batch: {len(pdb_ids)}")
        break  # print only the first batch then exit
