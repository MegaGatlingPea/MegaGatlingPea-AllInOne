import os
import random
import lmdb
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch

class MetaDataset(Dataset):
    def __init__(self, cluster_data_dir, k_shot, k_query):
        """
        Initialize MetaDataset, for meta-learning.
        
        Args:
            cluster_data_dir (str): directory path containing all cluster LMDB files.
            k_shot (int): number of samples in support set.
            k_query (int): number of samples in query set.
        """
        self.cluster_data_dir = cluster_data_dir
        self.k_shot = k_shot
        self.k_query = k_query

        # Get all LMDB paths
        self.lmdb_paths = self._get_all_lmdb_paths()

        # Map cluster_id to LMDB path
        self.cluster_id_to_lmdb_path = self._map_cluster_ids_to_lmdb_path()

        # Open all LMDB environments and keep them persistent
        self.envs = {cluster_id: lmdb.open(path, readonly=True, lock=False) 
                     for cluster_id, path in self.cluster_id_to_lmdb_path.items()}

        # Get PDB IDs list for each cluster
        self.cluster_ids = list(self.cluster_id_to_lmdb_path.keys())
        self.cluster_id_to_pdb_ids = self._get_cluster_pdb_ids()

    def _get_all_lmdb_paths(self):
        """Get all LMDB paths in cluster_data_dir."""
        lmdb_files = []
        for fname in os.listdir(self.cluster_data_dir):
            path = os.path.join(self.cluster_data_dir, fname)
            if os.path.isdir(path) and fname.endswith('.lmdb'):
                lmdb_files.append(path)
            elif os.path.isfile(path) and fname.endswith('.lmdb'):
                lmdb_files.append(path)
        return lmdb_files

    def _map_cluster_ids_to_lmdb_path(self):
        """Map cluster_id to LMDB path."""
        cluster_id_to_lmdb_path = {}
        for lmdb_path in self.lmdb_paths:
            basename = os.path.basename(lmdb_path)
            # Assuming the naming convention is cluster_{cluster_id}.lmdb
            if basename.startswith('cluster_') and basename.endswith('.lmdb'):
                cluster_id = basename[len('cluster_'):-len('.lmdb')]
                cluster_id_to_lmdb_path[cluster_id] = lmdb_path
        return cluster_id_to_lmdb_path

    def _get_cluster_pdb_ids(self):
        """Get PDB IDs list for each cluster."""
        cluster_id_to_pdb_ids = {}
        for cluster_id, env in self.envs.items():
            pdb_ids = []
            with env.begin() as txn:
                cursor = txn.cursor()
                for key, _ in cursor:
                    pdb_id = key.decode()
                    pdb_ids.append(pdb_id)
            cluster_id_to_pdb_ids[cluster_id] = pdb_ids
        return cluster_id_to_pdb_ids

    def __len__(self):
        """Return the number of clusters (tasks)."""
        return len(self.cluster_ids)

    def __getitem__(self, idx):
        """
        Get a task (cluster), including support set and query set.

        Args:
            idx (int): index of the cluster.

        Returns:
            dict: a dictionary containing 'support_set' and 'query_set', with values as lists of data samples.
        """
        cluster_id = self.cluster_ids[idx]
        env = self.envs[cluster_id]
        pdb_ids = self.cluster_id_to_pdb_ids[cluster_id]

        # Ensure enough samples
        total_samples_needed = self.k_shot + self.k_query
        if len(pdb_ids) < total_samples_needed:
            raise ValueError(f"Not enough samples in cluster {cluster_id}")

        # Randomly select support set and query set PDB IDs
        pdb_ids_sampled = random.sample(pdb_ids, total_samples_needed)
        support_pdb_ids = pdb_ids_sampled[:self.k_shot]
        query_pdb_ids = pdb_ids_sampled[self.k_shot:]

        # Load support set data
        support_set = []
        with env.begin() as txn:
            for pdb_id in support_pdb_ids:
                data = txn.get(pdb_id.encode())
                if data is not None:
                    item = pickle.loads(data)
                    protein_graph = item['protein_graph']
                    ligand_graph = item['ligand_graph']
                    kd_value = torch.tensor([item['kd_value']], dtype=torch.float)
                    support_set.append((protein_graph, ligand_graph, kd_value, pdb_id))
                else:
                    raise KeyError(f"PDB ID {pdb_id} not found in LMDB.")

        # Load query set data
        query_set = []
        with env.begin() as txn:
            for pdb_id in query_pdb_ids:
                data = txn.get(pdb_id.encode())
                if data is not None:
                    item = pickle.loads(data)
                    protein_graph = item['protein_graph']
                    ligand_graph = item['ligand_graph']
                    kd_value = torch.tensor([item['kd_value']], dtype=torch.float)
                    query_set.append((protein_graph, ligand_graph, kd_value, pdb_id))
                else:
                    raise KeyError(f"PDB ID {pdb_id} not found in LMDB.")

        return {'support_set': support_set, 'query_set': query_set}

    def close_envs(self):
        """Close all LMDB environments."""
        for env in self.envs.values():
            env.close()

    def __del__(self):
        """Ensure LMDB environments are closed upon deletion."""
        self.close_envs()

def meta_collate_fn(batch):
    """
    Custom collate function to batch multiple tasks.

    Args:
        batch (list): Each element is a dict with 'support_set' and 'query_set'.

    Returns:
        dict: Batched 'support_set' and 'query_set'.
    """
    batched_support_protein_graphs = []
    batched_support_ligand_graphs = []
    batched_support_kd_values = []
    batched_query_protein_graphs = []
    batched_query_ligand_graphs = []
    batched_query_kd_values = []

    for task in batch:
        support_set = task['support_set']
        query_set = task['query_set']

        # Support set
        for sample in support_set:
            protein_graph, ligand_graph, kd_value, pdb_id = sample
            batched_support_protein_graphs.append(protein_graph)
            batched_support_ligand_graphs.append(ligand_graph)
            batched_support_kd_values.append(kd_value)

        # Query set
        for sample in query_set:
            protein_graph, ligand_graph, kd_value, pdb_id = sample
            batched_query_protein_graphs.append(protein_graph)
            batched_query_ligand_graphs.append(ligand_graph)
            batched_query_kd_values.append(kd_value)

    # Batch the graphs using PyTorch Geometric's Batch
    batched_support_protein_graphs = Batch.from_data_list(batched_support_protein_graphs)
    batched_support_ligand_graphs = Batch.from_data_list(batched_support_ligand_graphs)
    batched_query_protein_graphs = Batch.from_data_list(batched_query_protein_graphs)
    batched_query_ligand_graphs = Batch.from_data_list(batched_query_ligand_graphs)

    # Stack KD values
    batched_support_kd_values = torch.cat(batched_support_kd_values, dim=0)
    batched_query_kd_values = torch.cat(batched_query_kd_values, dim=0)

    return {
        'support_set': {
            'protein_graphs': batched_support_protein_graphs,
            'ligand_graphs': batched_support_ligand_graphs,
            'kd_values': batched_support_kd_values
        },
        'query_set': {
            'protein_graphs': batched_query_protein_graphs,
            'ligand_graphs': batched_query_ligand_graphs,
            'kd_values': batched_query_kd_values
        }
    }

if __name__ == '__main__':
    cluster_data_dir = './../cluster_data'
    k_shot = 2
    k_query = 2
    meta_batch_size = 4
    num_workers = 1

    try:
        dataset = MetaDataset(cluster_data_dir=cluster_data_dir, k_shot=k_shot, k_query=k_query)
    except Exception as e:
        print(f"Error initializing MetaDataset: {e}")
        exit(1)

    try:
        dataloader = DataLoader(
            dataset,
            batch_size=meta_batch_size,
            shuffle=True,
            collate_fn=meta_collate_fn,
            num_workers=num_workers,
            pin_memory=True
        )
    except Exception as e:
        print(f"Error creating DataLoader: {e}")
        dataset.close_envs()
        exit(1)

    try:
        for batch_idx, batch in enumerate(dataloader):
            print(f"Batch {batch_idx + 1}:")
            
            support_set = batch['support_set']
            query_set = batch['query_set']
            
            print("  Support Set:")
            print(f"    Protein Graphs: {support_set['protein_graphs']}")
            print(f"    Ligand Graphs: {support_set['ligand_graphs']}")
            print(f"    KD Values: {support_set['kd_values'].shape}")
            
            print("  Query Set:")
            print(f"    Protein Graphs: {query_set['protein_graphs']}")
            print(f"    Ligand Graphs: {query_set['ligand_graphs']}")
            print(f"    KD Values: {query_set['kd_values'].shape}")
            
            break
    except Exception as e:
        print(f"Error during data loading: {e}")
    finally:
        dataset.close_envs()
