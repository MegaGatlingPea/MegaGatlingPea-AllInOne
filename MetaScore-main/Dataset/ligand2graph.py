import torch
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np

def create_feature_mappings():
    atom_feature_mapping = {
        'atomic_num': {num: idx for idx, num in enumerate(range(1, 119))},  # 1-118
        'chirality': {0: 0, 1: 1, 2: 2},  # ChiralityType mapping
        'degree': {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6},
        'formal_charge': {-3: 0, -2: 1, -1: 2, 0: 3, 1: 4, 2: 5, 3: 6},
        'num_h': {0: 0, 1: 1, 2: 2, 3: 3, 4: 4},
        'number_radical_e': {0: 0, 1: 1, 2: 2},
        'hybridization': {Chem.rdchem.HybridizationType.SP: 0, 
                          Chem.rdchem.HybridizationType.SP2: 1, 
                          Chem.rdchem.HybridizationType.SP3: 2,
                          Chem.rdchem.HybridizationType.SP3D: 3,
                          Chem.rdchem.HybridizationType.SP3D2: 4},
        'is_aromatic': {False: 0, True: 1},
        'is_in_ring': {False: 0, True: 1}
    }
    
    bond_feature_mapping = {
        'bond_type': {Chem.rdchem.BondType.SINGLE: 0, 
                      Chem.rdchem.BondType.DOUBLE: 1, 
                      Chem.rdchem.BondType.TRIPLE: 2, 
                      Chem.rdchem.BondType.AROMATIC: 3},
        'bond_stereo': {Chem.rdchem.BondStereo.STEREONONE: 0,
                        Chem.rdchem.BondStereo.STEREOZ: 1,
                        Chem.rdchem.BondStereo.STEREOE: 2,
                        Chem.rdchem.BondStereo.STEREOCIS: 3,
                        Chem.rdchem.BondStereo.STEREOTRANS: 4},
        'is_conjugated': {False: 0, True: 1}
    }
    
    return atom_feature_mapping, bond_feature_mapping

def ligand2graph(sdf_file, remove_hs=True):
    # Read SDF file without sanitization, optionally removing hydrogens
    suppl = Chem.SDMolSupplier(sdf_file, sanitize=False, removeHs=remove_hs)
    mol = next(suppl)
    if mol is None:
        raise ValueError(f"Cannot read SDF file: {sdf_file}")
    
    # Create feature mappings
    atom_feature_mapping, bond_feature_mapping = create_feature_mappings()
    
    # Extract atom features
    atom_features = []
    for atom in mol.GetAtoms():
        features = [
            atom_feature_mapping['atomic_num'].get(atom.GetAtomicNum(), len(atom_feature_mapping['atomic_num']) - 1),
            atom_feature_mapping['chirality'][int(atom.GetChiralTag())],
            atom_feature_mapping['degree'].get(atom.GetTotalDegree(), len(atom_feature_mapping['degree']) - 1),
            atom_feature_mapping['formal_charge'].get(atom.GetFormalCharge(), len(atom_feature_mapping['formal_charge']) - 1),
            atom_feature_mapping['num_h'].get(atom.GetTotalNumHs(), len(atom_feature_mapping['num_h']) - 1),
            atom_feature_mapping['number_radical_e'].get(atom.GetNumRadicalElectrons(), len(atom_feature_mapping['number_radical_e']) - 1),
            atom_feature_mapping['hybridization'].get(atom.GetHybridization(), len(atom_feature_mapping['hybridization']) - 1),
            atom_feature_mapping['is_aromatic'][atom.GetIsAromatic()],
            atom_feature_mapping['is_in_ring'][atom.IsInRing()]
        ]
        atom_features.append(features)
    
    # Extract edge information and build edge indices
    edge_indices = []
    edge_attributes = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_indices += [[i, j], [j, i]]
        
        bond_features = [
            bond_feature_mapping['bond_type'][bond.GetBondType()],
            bond_feature_mapping['bond_stereo'][bond.GetStereo()],
            bond_feature_mapping['is_conjugated'][bond.GetIsConjugated()]
        ]
        edge_attributes += [bond_features] * 2
    
    # Get 3D coordinates
    conf = mol.GetConformer()
    positions = conf.GetPositions()
    
    # Convert to PyTorch tensors
    x = torch.tensor(atom_features, dtype=torch.long)
    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attributes, dtype=torch.long)
    pos = torch.tensor(positions, dtype=torch.float)
    
    # Create Data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos)
    
    return data

# Example usage
if __name__ == "__main__":
    sdf_file = './../../Testset/1afk/1afk_ligand.sdf'
    ligand_graph = ligand2graph(sdf_file)
    print(f"Number of nodes: {ligand_graph.num_nodes}")
    print(f"Number of edges: {ligand_graph.num_edges}")
    print(f"Node feature shape: {ligand_graph.x.shape}")
    print(f"Edge feature shape: {ligand_graph.edge_attr.shape}")
    print(f"Position shape: {ligand_graph.pos.shape}")