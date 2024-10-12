import csv
import pickle
import subprocess
import tempfile
import os
from rdkit import Chem
from rdkit.Chem import AllChem
from ligand2graph import mol2graph

def read_smiles_from_csv(csv_file, smiles_column='smiles'):
    """
    读取CSV文件中的SMILES列
    """
    smiles_list = []
    with open(csv_file, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            smiles = row.get(smiles_column)
            if smiles:
                smiles_list.append(smiles)
    return smiles_list

def generate_mol_with_obabel(smiles):
    """
    使用Open Babel通过命令行生成带有3D构象的mol对象
    """
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".smi") as tmp_input:
            tmp_input.write(smiles.encode())
            tmp_input_path = tmp_input.name

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mol") as tmp_output:
            tmp_output_path = tmp_output.name

        # 调用obabel生成3D构象
        subprocess.run([
            "obabel",
            tmp_input_path,
            "-O",
            tmp_output_path,
            "--gen3d",
            "--force"
        ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # 读取生成的mol文件
        mol = Chem.MolFromMolFile(tmp_output_path, removeHs=False)
        if mol is None:
            raise ValueError("Open Babel生成的mol文件无法解析")

        return mol
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Open Babel处理失败: {e.stderr.decode().strip()}")
    finally:
        # 清理临时文件
        if os.path.exists(tmp_input_path):
            os.remove(tmp_input_path)
        if os.path.exists(tmp_output_path):
            os.remove(tmp_output_path)

def smiles_to_mol(smiles, max_iter=1000):
    """
    将SMILES字符串转换为RDKit mol对象
    并生成3D构象及进行力场优化
    尝试多种嵌入和优化方法，必要时使用Open Babel作为备选方案
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("解析错误")

    # 生成3D构象
    if mol.GetNumConformers() == 0:
        # 尝试多种嵌入方法
        embed_methods = [
            lambda m: AllChem.EmbedMolecule(m, AllChem.ETKDG()),
            lambda m: AllChem.EmbedMolecule(m, AllChem.ETKDG(), randomSeed=0xf00d)
        ]
        embed_success = False
        for method in embed_methods:
            result = method(mol)
            if result == 0:
                embed_success = True
                print("构象嵌入成功")
                break
        if not embed_success:
            print("RDKit嵌入失败，尝试使用Open Babel")
            mol = generate_mol_with_obabel(smiles)
            if mol is None:
                raise ValueError("构象嵌入失败")
            print("Open Babel构象嵌入成功")

    # 优化构象，增加迭代次数
    optimization_methods = [
        lambda m: AllChem.MMFFOptimizeMolecule(m, maxIters=max_iter, mmffVariant='MMFF94'),
        lambda m: AllChem.UFFOptimizeMolecule(m, maxIters=max_iter)
    ]
    optimize_success = False
    for method in optimization_methods:
        result = method(mol)
        if result == 0:
            optimize_success = True
            print("构象优化成功")
            break
        else:
            print("当前优化方法失败，尝试下一个方法")
    if not optimize_success:
        print("RDKit优化失败，尝试使用Open Babel优化")
        try:
            # 使用Open Babel进行优化
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mol") as tmp_mol:
                Chem.MolToMolFile(mol, tmp_mol.name)
                tmp_mol_path = tmp_mol.name

            tmp_optimized_mol_path = tmp_mol_path.replace(".mol", "_optimized.mol")

            subprocess.run([
                "obabel",
                tmp_mol_path,
                "-O",
                tmp_optimized_mol_path,
                "--minimize",
                "--force"
            ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            optimized_mol = Chem.MolFromMolFile(tmp_optimized_mol_path, removeHs=False)
            if optimized_mol is None:
                raise ValueError("Open Babel优化后的mol文件无法解析")

            mol = optimized_mol
            optimize_success = True
            print("Open Babel优化成功")
        except subprocess.CalledProcessError as e:
            raise ValueError("构象优化失败")
        finally:
            # 清理临时文件
            if os.path.exists(tmp_mol_path):
                os.remove(tmp_mol_path)
            if os.path.exists(tmp_optimized_mol_path):
                os.remove(tmp_optimized_mol_path)

    return mol

def process_smiles(csv_file, output_pkl, failed_list_file, smiles_column='smiles'):
    """
    处理CSV文件中的SMILES，并将生成的图保存到Pickle文件
    同时记录成功和失败的分子
    """
    smiles_list = read_smiles_from_csv(csv_file, smiles_column)
    graph_dict = {}
    failed_list = []

    for idx, smiles in enumerate(smiles_list):
        try:
            mol = smiles_to_mol(smiles)
            graph = mol2graph(mol, remove_hs=True, offset=128)
            graph_dict[idx] = graph
            print(f"成功处理索引 {idx}: SMILES={smiles}")
        except ValueError as ve:
            error_type = str(ve)
            if "构象嵌入失败" in error_type:
                failure_reason = "嵌入失败"
            elif "构象优化失败" in error_type:
                failure_reason = "优化失败"
            else:
                failure_reason = "其他错误"
            print(f"处理失败索引 {idx}: SMILES={smiles}, 原因: {failure_reason}")
            failed_list.append({'index': idx, 'smiles': smiles, 'error': failure_reason})
        except Exception as e:
            # 为了调试，临时记录完整错误信息
            print(f"处理失败索引 {idx}: SMILES={smiles}, 原因: {e}")
            failed_list.append({'index': idx, 'smiles': smiles, 'error': "其他错误"})

    # 保存图到Pickle文件
    with open(output_pkl, 'wb') as f:
        pickle.dump(graph_dict, f)
    print(f"所有成功的图已保存到 {output_pkl}")

    # 保存失败列表到CSV文件
    with open(failed_list_file, 'w', newline='') as f:
        fieldnames = ['index', 'smiles', 'error']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in failed_list:
            writer.writerow(item)
    print(f"所有失败的分子已保存到 {failed_list_file}")

if __name__ == "__main__":
    import argparse
    import shutil
    import logging

    # 检查obabel是否安装
    if shutil.which("obabel") is None:
        raise EnvironmentError("Open Babel未安装或obabel命令不可用，请安装Open Babel。")

    # 配置日志记录
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    parser = argparse.ArgumentParser(description="将CSV中的SMILES转换为图并保存为Pickle文件，同时记录失败的分子")
    parser.add_argument('--csv', type=str, help='输入CSV文件路径', default='./../raw/test.csv')
    parser.add_argument('--output', type=str, help='输出Pickle文件路径', default='./../processed/test.pkl')
    parser.add_argument('--failed', type=str, help='输出失败列表文件路径', default='./../processed/failed_list.csv')
    parser.add_argument('--smiles_col', type=str, default='smiles', help='CSV中的SMILES列名')

    args = parser.parse_args()

    process_smiles(args.csv, args.output, args.failed, args.smiles_col)