import torch
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np

def ligand2graph(sdf_file):
    # Read SDF file
    mol = Chem.SDMolSupplier(sdf_file, removeHs=False)[0]
    if mol is None:
        raise ValueError(f"Cannot read SDF file: {sdf_file}")

    # Extract atom features
    atom_features = []
    for atom in mol.GetAtoms():
        features = [
            atom.GetAtomicNum(),
            atom.GetChiralTag(),
            atom.GetDegree(),
            atom.GetFormalCharge(),
            atom.GetNumExplicitHs(),
            atom.GetNumRadicalElectrons(),
            atom.GetHybridization(),
            atom.GetIsAromatic(),
            atom.IsInRing()
        ]
        atom_features.append(features)

    # Extract edge information and build edge indices
    edge_indices = []
    edge_attributes = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_indices += [[i, j], [j, i]]
        
        bond_type = bond.GetBondType()
        bond_stereo = bond.GetStereo()
        is_conjugated = bond.GetIsConjugated()
        
        edge_attributes += [[bond_type, bond_stereo, is_conjugated]] * 2

    # Get 3D coordinates
    conf = mol.GetConformer()
    positions = conf.GetPositions()

    # Convert to PyTorch tensors
    x = torch.tensor(atom_features, dtype=torch.float)
    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attributes, dtype=torch.float)
    pos = torch.tensor(positions, dtype=torch.float)

    # Create Data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos)

    return data