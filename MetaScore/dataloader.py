import lmdb
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch

class PDBBindDataset(Dataset):
    def __init__(self, lmdb_path="./../Testset/pdbbind.lmdb"):
        self.env = lmdb.open(lmdb_path, readonly=True, lock=False)
        self.txn = self.env.begin()
        self.data_dict = self._load_data()

    def _load_data(self):
        data_dict = {}
        for key, value in self.txn.cursor():
            cluster_data = pickle.loads(value)
            for pdb_id, pdb_data in cluster_data.items():
                data_dict[pdb_id] = pdb_data
        return data_dict

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, idx):
        pdb_id = list(self.data_dict.keys())[idx]
        item = self.data_dict[pdb_id]
        
        protein_graph = item['protein_graph']
        ligand_graph = item['ligand_graph']
        kd_value = torch.tensor([item['kd_value']], dtype=torch.float)
        
        return protein_graph, ligand_graph, kd_value, pdb_id

    def __del__(self):
        self.env.close()

def collate_fn(batch):
    protein_graphs, ligand_graphs, kd_values, pdb_ids = zip(*batch)
    
    batched_protein_graphs = Batch.from_data_list(protein_graphs)
    batched_ligand_graphs = Batch.from_data_list(ligand_graphs)
    batched_kd_values = torch.cat(kd_values)
    
    return batched_protein_graphs, batched_ligand_graphs, batched_kd_values, pdb_ids

# usage 
# lmdb_path = "./../Testset/pdbbind.lmdb"
# dataset = PDBBindDataset()
# dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)