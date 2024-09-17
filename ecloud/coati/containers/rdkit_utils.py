#
# Any useful rdkit functions are sequestered to this file.
# to keep minimal dependence on it. Instead it imports stuff from here
#
import functools
import random
from operator import itemgetter
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import rdkit
from rdkit import Chem, DataStructs
from rdkit.Chem import (
    Crippen,
    Descriptors,
    Draw,
    Lipinski,
    PandasTools,
    rdMolDescriptors,
)
from rdkit.Chem.AllChem import (
    EmbedMolecule,
    EmbedMultipleConfs,
    GetMorganFingerprintAsBitVect,
)
from rdkit.Chem.MolStandardize.rdMolStandardize import Uncharger
from rdkit.Chem.rdForceFieldHelpers import MMFFOptimizeMoleculeConfs
from rdkit.Chem.SaltRemover import SaltRemover


def works_on_smiles(raise_on_failure: bool):
    """
    装饰器函数，将处理RDKit分子对象的函数转换为可以直接处理SMILES字符串的函数。
    
    参数:
    raise_on_failure (bool): 决定在转换失败时是抛出异常还是返回None
    
    功能:
    - 尝试将SMILES字符串转换为RDKit分子对象
    - 调用原始函数并处理结果
    - 如果原始函数返回RDKit分子对象，将其转换回SMILES字符串
    - 处理元组返回值，将其中的分子对象转换为SMILES
    - 提供错误处理机制
    """

    def decorator(mol_func):
        @functools.wraps(mol_func)
        def wrapped_func(*args, **kwargs):
            if isinstance(args[0], str):
                mol = Chem.MolFromSmiles(args[0])
                if mol!= None:
                    new_args = list(args)
                    new_args[0] = mol
                    try:
                        results = mol_func(*new_args, **kwargs)
                    except Exception as Ex:
                        if raise_on_failure:
                            raise Ex
                        else:
                            print(f"Exception: {Ex} for smiles: {args[0]}")
                            return None
                    # try to convert back to smiles...
                    if isinstance(results, Chem.Mol):
                        return Chem.MolToSmiles(results)
                    elif isinstance(results, tuple):
                        return tuple(
                            Chem.MolToSmiles(res) if isinstance(res, Chem.Mol) else res
                            for res in results
                        )
                    else:
                        return results
                else:
                    if raise_on_failure:
                        raise ValueError(f"{args[0]} could not be converted to mol.")
                    else:
                        return None
            else:
                return mol_func(*args, **kwargs)

        return wrapped_func

    return decorator


def rdkit_version():
    """
    返回当前使用的RDKit版本。
    
    返回:
    str: RDKit版本号
    """
    return rdkit.__version__


def canon_smiles(s):
    """
    将输入的SMILES字符串转换为规范化的SMILES表示。
    
    参数:
    s (str): 输入的SMILES字符串
    
    返回:
    str: 规范化的SMILES字符串，或在转换失败时返回"BAD_SMILES"
    
    功能:
    - 将输入转换为RDKit分子对象
    - 进行Kekulize处理（将芳香结构转换为单双键表示）
    - 返回规范化的SMILES字符串
    """
    try:
        m = Chem.MolFromSmiles(s)
        if not m is None:
            Chem.Kekulize(m)
            return Chem.MolToSmiles(m)
        else:
            return "BAD_SMILES"
    except:
        return "BAD_SMILES"


@works_on_smiles(raise_on_failure=True)
def sim_mol(mol1: Chem.Mol, mol2: Chem.Mol) -> float:
    """
    计算两个分子之间的相似度。
    
    参数:
    mol1, mol2 (Chem.Mol): 要比较的两个分子
    
    返回:
    float: 两个分子之间的Tanimoto相似度（0到1之间）
    
    功能:
    - 使用ECFP4（扩展连接指纹）生成分子指纹
    - 计算Tanimoto相似度
    - 可以直接接受SMILES字符串作为输入
    """
    fp1 = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol1, 2, 2048)
    fp2 = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol2, 2, 2048)
    return DataStructs.TanimotoSimilarity(fp1, fp2)


def identical_canonsmi(smi1: str, smi2: str, use_chiral: int = 1) -> bool:
    """
    判断两个SMILES字符串是否表示相同的分子。
    
    参数:
    smi1, smi2 (str): 要比较的两个SMILES字符串
    use_chiral (int): 是否考虑手性，默认为1（考虑）
    
    返回:
    bool: 如果两个SMILES表示相同的分子则返回True，否则返回False
    
    功能:
    - 使用RDKit的CanonSmiles函数生成规范化的SMILES
    - 比较两个规范化的SMILES是否完全相同
    """
    return Chem.CanonSmiles(smi1, useChiral=use_chiral) == Chem.CanonSmiles(
        smi2, useChiral=use_chiral
    )


@works_on_smiles(raise_on_failure=True)
def draw_mol(mol: Chem.Mol, size=(300, 300)):
    """
    绘制分子的2D结构图。
    
    参数:
    mol (Chem.Mol): 要绘制的分子
    size (tuple): 图像大小，默认为(300, 300)
    
    返回:
    Image: 分子的2D结构图像
    
    功能:
    - 使用RDKit的Draw.MolToImage函数绘制分子
    - 可以指定图像大小
    - 可以接受SMILES字符串作为输入
    """
    return Draw.MolToImage(mol, size=size)


def permute_smiles(smiles: str) -> str:
    """
    生成给定SMILES的一个随机排列版本。
    
    参数:
    smiles (str): 输入的SMILES字符串
    
    返回:
    str: 重新排列后的非规范化SMILES字符串
    
    功能:
    - 将SMILES转换为分子对象
    - 生成原子的随机排列
    - 使用Chem.RenumberAtoms重新编号原子
    - 返回重新编号后的非规范化SMILES
    """
    mol = Chem.MolFromSmiles(smiles)
    ans = list(range(mol.GetNumAtoms()))
    random.shuffle(ans)
    nm = Chem.RenumberAtoms(mol, ans)
    return Chem.MolToSmiles(nm, canonical=False)


def draw_smi_grid(
    smis: List[str], mols_per_row=5, sub_img_size=(300, 300), legends=None
):
    """
    绘制多个分子的2D结构图网格。

    参数:
    smis (List[str]): SMILES字符串列表
    mols_per_row (int): 每行显示的分子数量，默认为5
    sub_img_size (tuple): 每个分子图像的大小，默认为(300, 300)
    legends (List[str], optional): 每个分子的标签列表

    返回:
    Image: 包含多个分子2D结构的网格图像

    功能:
    - 将SMILES字符串列表转换为分子对象
    - 使用RDKit的Draw.MolsToGridImage函数绘制分子网格
    - 可以自定义每行分子数、图像大小和标签
    """
    return Draw.MolsToGridImage(
        [Chem.MolFromSmiles(smi) for smi in smis],
        molsPerRow=mols_per_row,
        subImgSize=sub_img_size,
        legends=legends,
    )


def disable_logger():
    """
    禁用RDKit的日志输出。

    功能:
    - 使用RDKit的RDLogger.DisableLog函数禁用所有rdApp相关的日志
    """
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")


@works_on_smiles(raise_on_failure=True)
def mol_to_morgan(
    mol: Chem.Mol,
    radius: int = 3,
    n_bits: int = 2048,
    chiral: bool = False,
    features: bool = False,
) -> np.ndarray:
    """
    生成分子的Morgan指纹。

    参数:
    mol (Chem.Mol): 输入分子
    radius (int): Morgan指纹的半径，默认为3
    n_bits (int): 指纹的位数，默认为2048
    chiral (bool): 是否考虑手性，默认为False
    features (bool): 是否使用特征化的Morgan指纹，默认为False

    返回:
    np.ndarray: 表示Morgan指纹的numpy数组

    功能:
    - 使用RDKit的GetMorganFingerprintAsBitVect函数生成Morgan指纹
    - 将位向量转换为numpy数组
    - 可以直接接受SMILES字符串作为输入
    """
    return np.frombuffer(
        GetMorganFingerprintAsBitVect(
            mol,
            radius=radius,
            nBits=n_bits,
            useChirality=chiral,
            useFeatures=features,
        )
        .ToBitString()
        .encode(),
        "u1",
    ) - ord("0")


@works_on_smiles(raise_on_failure=False)
def mol_to_atoms_coords(
    m: Chem.Mol,
    hydrogenate: bool = True,
    adj_matrix: bool = False,
    do_morgan: bool = False,
    optimize: bool = False,
    numConfs: int = 1,
    numThreads: int = 1,
):
    """
    生成分子的原子坐标和其他相关信息。

    参数:
    m (Chem.Mol): 输入分子
    hydrogenate (bool): 是否添加氢原子，默认为True
    adj_matrix (bool): 是否返回邻接矩阵，默认为False
    do_morgan (bool): 是否计算Morgan指纹，默认为False
    optimize (bool): 是否优化分子构象，默认为False
    numConfs (int): 生成的构象数量，默认为1
    numThreads (int): 使用的线程数，默认为1

    返回:
    tuple: 包含原子类型、坐标和其他可选信息的元组

    功能:
    - 添加氢原子（如果指定）
    - 生成3D构象
    - 可选地进行MMFF优化
    - 提取原子类型和坐标
    - 可选地生成邻接矩阵和Morgan指纹
    - 可以直接接受SMILES字符串作为输入
    """

    m3 = Chem.AddHs(m) if hydrogenate else m
    if optimize and hydrogenate:
        try:
            EmbedMultipleConfs(
                m3,
                randomSeed=0xF00D,
                numConfs=numConfs,
                pruneRmsThresh=0.125,
                ETversion=1,
                numThreads=numThreads,
            )
            opt = MMFFOptimizeMoleculeConfs(
                m3, mmffVariant="MMFF94s", numThreads=numThreads, maxIters=10000
            )
            opt = np.array(opt)
            converged = opt[:, 0] == 0
            lowest_eng_conformer = np.argmin(opt[converged][:, 1])
            lowest_energy = opt[converged][lowest_eng_conformer, 1]
            best_conf = np.arange(opt.shape[0])[converged][lowest_eng_conformer]
            c0 = m3.GetConformer(id=int(best_conf))
        except Exception as Ex:
            EmbedMolecule(m3, randomSeed=0xF00D)
            c0 = m3.GetConformers()[-1]
            lowest_energy = None
    else:
        EmbedMolecule(m3, randomSeed=0xF00D)
        c0 = m3.GetConformers()[-1]
    coords = c0.GetPositions()
    atoms = np.array([X.GetAtomicNum() for X in m3.GetAtoms()], dtype=np.uint8)

    to_return = [atoms, coords]

    if adj_matrix:
        to_return.append(Chem.GetAdjacencyMatrix(m3))

    # NOT using the relaxed/confgen'd molecule with HS - rdkit
    # is surprisingly sensitive to this.
    if do_morgan:
        to_return.append(mol_to_morgan(m, radius=3, n_bits=2048, chiral=False))

    if optimize:
        to_return.append(lowest_energy)

    return tuple(to_return)


def read_sdf(sdf: Any) -> pd.DataFrame:
    """
    读取SDF文件并转换为pandas DataFrame。

    参数:
    sdf (Any): SDF文件路径或文件对象

    返回:
    pd.DataFrame: 包含SDF数据的DataFrame

    功能:
    - 使用RDKit的PandasTools.LoadSDF函数读取SDF文件
    - 将SDF数据转换为pandas DataFrame格式
    """
    return PandasTools.LoadSDF(sdf, smilesName="SMILES")


@works_on_smiles(raise_on_failure=False)
def mol_standardize(mol: Chem.Mol) -> Optional[Chem.Mol]:
    """
    标准化分子结构。

    参数:
    mol (Chem.Mol): 输入分子

    返回:
    Optional[Chem.Mol]: 标准化后的分子，如果失败则返回None

    功能:
    - 移除盐
    - 选择最大的分子片段
    - 中和分子电荷
    - 可以直接接受SMILES字符串作为输入
    """
    # salt removal
    salt_remover = SaltRemover()
    res_mol = salt_remover.StripMol(mol, dontRemoveEverything=True)

    # # uncharged mol

    if res_mol.GetNumAtoms():
        # largest component
        frag_list = list(Chem.GetMolFrags(res_mol, asMols=True))
        frag_mw_list = [(x.GetNumAtoms(), x) for x in frag_list]
        frag_mw_list.sort(key=itemgetter(0), reverse=True)
        # neutralize fragment
        if len(frag_mw_list) > 0:
            return Uncharger().uncharge(frag_mw_list[0][1])
        return None
    else:
        print(f'Failed salt removal: "{Chem.MolToSmiles(mol)}"')
        return None


@works_on_smiles(raise_on_failure=False)
def mol_properties(mol: Chem.Mol) -> Dict[str, Any]:
    """
    计算分子的各种物理化学性质。

    参数:
    mol (Chem.Mol): 输入分子

    返回:
    Dict[str, Any]: 包含各种分子性质的字典

    功能:
    - 计算分子量、极性表面积、sp3碳比例等多种性质
    - 使用RDKit的Descriptors和Lipinski模块计算各种分子描述符
    - 可以直接接受SMILES字符串作为输入
    """
    return {
        "MolWt": Descriptors.MolWt(mol),
        "TPSA": Descriptors.TPSA(mol),
        "FractionCSP3": Lipinski.FractionCSP3(mol),
        "HeavyAtomCount": Lipinski.HeavyAtomCount(mol),
        "NumAliphaticRings": Lipinski.NumAliphaticRings(mol),
        "NumAromaticRings": Lipinski.NumAromaticRings(mol),
        "NumHAcceptors": Lipinski.NumHAcceptors(mol),
        "NumHDonors": Lipinski.NumHDonors(mol),
        "NumHeteroatoms": Lipinski.NumHeteroatoms(mol),
        "NumRotatableBonds": Lipinski.NumRotatableBonds(mol),
        "NumSaturatedRings": Lipinski.NumSaturatedRings(mol),
        "RingCount": Lipinski.RingCount(mol),
        "MolLogP": Crippen.MolLogP(mol),
    }
