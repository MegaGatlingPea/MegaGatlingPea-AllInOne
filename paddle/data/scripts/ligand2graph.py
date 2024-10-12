import torch
from torch_geometric.data import Data
from rdkit import Chem
import numpy as np

# 定义允许的特征
allowable_features = {
    'possible_atomic_num_list': list(range(1, 119)) + ['misc'],
    'possible_chirality_list': [
        'CHI_UNSPECIFIED',
        'CHI_TETRAHEDRAL_CW',
        'CHI_TETRAHEDRAL_CCW',
        'CHI_SQUAREPLANAR',
        'CHI_OCTAHEDRAL',
        'CHI_TRIGONALBIPYRAMIDAL',
        'CHI_OTHER'
    ],
    'possible_degree_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],
    'possible_formal_charge_list': [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 'misc'],
    'possible_numH_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
    'possible_number_radical_e_list': [0, 1, 2, 3, 4, 'misc'],
    'possible_hybridization_list': [
        'UNSPECIFIED', 'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'misc'
    ],
    'possible_is_aromatic_list': [False, True],
    'possible_is_in_ring_list': [False, True],
    'possible_bond_type_list': [
        'SINGLE',
        'DOUBLE',
        'TRIPLE',
        'AROMATIC',
        'misc'
    ],
    'possible_bond_stereo_list': [
        'STEREONONE',
        'STEREOZ',
        'STEREOE',
        'STEREOCIS',
        'STEREOTRANS',
        'STEREOANY',
    ],
    'possible_is_conjugated_list': [False, True],
}

def safe_index(l, e):
    """
    返回元素 e 在列表 l 中的索引。如果不存在，返回最后一个索引
    """
    try:
        return l.index(e)
    except ValueError:
        return len(l) - 1

def atom_to_feature_vector(atom):
    """
    将 RDKit 原子对象转换为特征索引列表
    """
    atom_feature = [
        safe_index(allowable_features['possible_atomic_num_list'], atom.GetAtomicNum()),
        allowable_features['possible_chirality_list'].index(str(atom.GetChiralTag())),
        safe_index(allowable_features['possible_degree_list'], atom.GetTotalDegree()),
        safe_index(allowable_features['possible_formal_charge_list'], atom.GetFormalCharge()),
        safe_index(allowable_features['possible_numH_list'], atom.GetTotalNumHs()),
        safe_index(allowable_features['possible_number_radical_e_list'], atom.GetNumRadicalElectrons()),
        safe_index(allowable_features['possible_hybridization_list'], str(atom.GetHybridization())),
        allowable_features['possible_is_aromatic_list'].index(atom.GetIsAromatic()),
        allowable_features['possible_is_in_ring_list'].index(atom.IsInRing()),
    ]
    return atom_feature

def bond_to_feature_vector(bond):
    """
    将 RDKit 键对象转换为特征索引列表
    """
    bond_feature = [
        safe_index(allowable_features['possible_bond_type_list'], str(bond.GetBondType())),
        allowable_features['possible_bond_stereo_list'].index(str(bond.GetStereo())),
        allowable_features['possible_is_conjugated_list'].index(bond.GetIsConjugated()),
    ]
    return bond_feature

def get_atom_feature_dims():
    return list(map(len, [
        allowable_features['possible_atomic_num_list'],
        allowable_features['possible_chirality_list'],
        allowable_features['possible_degree_list'],
        allowable_features['possible_formal_charge_list'],
        allowable_features['possible_numH_list'],
        allowable_features['possible_number_radical_e_list'],
        allowable_features['possible_hybridization_list'],
        allowable_features['possible_is_aromatic_list'],
        allowable_features['possible_is_in_ring_list']
    ]))

def get_bond_feature_dims():
    return list(map(len, [
        allowable_features['possible_bond_type_list'],
        allowable_features['possible_bond_stereo_list'],
        allowable_features['possible_is_conjugated_list']
    ]))

def atom_feature_vector_to_dict(atom_feature, offset=128):
    [atomic_num_idx, chirality_idx, degree_idx, formal_charge_idx,
     num_h_idx, number_radical_e_idx, hybridization_idx, is_aromatic_idx, is_in_ring_idx] = atom_feature

    feature_dict = {
        'atomic_num': allowable_features['possible_atomic_num_list'][atomic_num_idx % offset],
        'chirality': allowable_features['possible_chirality_list'][chirality_idx % offset],
        'degree': allowable_features['possible_degree_list'][degree_idx % offset],
        'formal_charge': allowable_features['possible_formal_charge_list'][formal_charge_idx % offset],
        'num_h': allowable_features['possible_numH_list'][num_h_idx % offset],
        'num_rad_e': allowable_features['possible_number_radical_e_list'][number_radical_e_idx % offset],
        'hybridization': allowable_features['possible_hybridization_list'][hybridization_idx % offset],
        'is_aromatic': allowable_features['possible_is_aromatic_list'][is_aromatic_idx % offset],
        'is_in_ring': allowable_features['possible_is_in_ring_list'][is_in_ring_idx % offset]
    }

    return feature_dict

def bond_feature_vector_to_dict(bond_feature, offset=128):
    [bond_type_idx, bond_stereo_idx, is_conjugated_idx] = bond_feature

    feature_dict = {
        'bond_type': allowable_features['possible_bond_type_list'][bond_type_idx % offset],
        'bond_stereo': allowable_features['possible_bond_stereo_list'][bond_stereo_idx % offset],
        'is_conjugated': allowable_features['possible_is_conjugated_list'][is_conjugated_idx % offset]
    }

    return feature_dict

def ligand2graph(sdf_file, remove_hs=True, offset=128):
    suppl = Chem.SDMolSupplier(sdf_file, sanitize=False, removeHs=remove_hs)
    mol = next(suppl)
    if mol is None:
        raise ValueError(f"Cannot read SDF file: {sdf_file}")

    # Extract atom features
    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append(atom_to_feature_vector(atom))

    # Extract edge information and build edge indices
    edge_indices = []
    edge_attributes = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_indices += [[i, j], [j, i]]
        
        bond_feature = bond_to_feature_vector(bond)
        edge_attributes += [bond_feature] * 2

    # Get 3D coordinates
    conf = mol.GetConformer()
    positions = conf.GetPositions()

    # Convert to PyTorch tensors
    x = torch.tensor(atom_features, dtype=torch.long)
    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attributes, dtype=torch.long)
    pos = torch.tensor(positions, dtype=torch.float)

    # Apply offset to separate feature spaces
    num_atom_features = len(get_atom_feature_dims())
    num_bond_features = len(get_bond_feature_dims())
    
    x = x + (torch.arange(num_atom_features) * offset).unsqueeze(0)
    edge_attr = edge_attr + (torch.arange(num_bond_features) * offset).unsqueeze(0)

    # Create Data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos)
    
    return data

def mol2graph(mol, remove_hs=True, offset=128):
    """
    将 RDKit mol 对象转换为 PyTorch Geometric 的 Data 对象
    """
    if remove_hs:
        mol = Chem.RemoveHs(mol)

    # 提取原子特征
    atom_features = [atom_to_feature_vector(atom) for atom in mol.GetAtoms()]

    # 提取键信息并构建边索引
    edge_indices = []
    edge_attributes = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_indices += [[i, j], [j, i]]
        
        bond_feature = bond_to_feature_vector(bond)
        edge_attributes += [bond_feature, bond_feature]

    # 获取 3D 坐标
    conf = mol.GetConformer()
    positions = conf.GetPositions()

    # 转换为 PyTorch 张量
    x = torch.tensor(atom_features, dtype=torch.long)
    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attributes, dtype=torch.long)
    pos = torch.tensor(positions, dtype=torch.float)

    # 应用偏移量以区分特征空间
    num_atom_features = len(get_atom_feature_dims())
    num_bond_features = len(get_bond_feature_dims())
    
    x = x + (torch.arange(num_atom_features) * offset).unsqueeze(0)
    edge_attr = edge_attr + (torch.arange(num_bond_features) * offset).unsqueeze(0)

    # 创建 Data 对象
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos)
    
    return data

# 示例用法
if __name__ == "__main__":
    from rdkit import Chem
    from rdkit.Chem import AllChem

    sdf_file = "./../../Testset/1afk/1afk_ligand.sdf"
    suppl = Chem.SDMolSupplier(sdf_file, sanitize=False, removeHs=True)
    mol = next(suppl)
    if mol is None:
        raise ValueError(f"无法读取 SDF 文件: {sdf_file}")

    # 生成3D构象
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    AllChem.MMFFOptimizeMolecule(mol, mmffVariant='MMFF94')

    ligand_graph = ligand2graph(mol)
    print(f"节点数量: {ligand_graph.num_nodes}")
    print(f"边数量: {ligand_graph.num_edges}")
    print(f"节点特征形状: {ligand_graph.x.shape}")
    print(f"边特征形状: {ligand_graph.edge_attr.shape}")
    print(f"位置形状: {ligand_graph.pos.shape}")

    # 解码一个节点特征以供检查
    if ligand_graph.num_nodes > 0:
        print("示例原子特征:")
        print(atom_feature_vector_to_dict(ligand_graph.x[0].tolist()))

    # 解码一个键特征以供检查
    if ligand_graph.num_edges > 0:
        print("示例键特征:")
        print(bond_feature_vector_to_dict(ligand_graph.edge_attr[0].tolist()))