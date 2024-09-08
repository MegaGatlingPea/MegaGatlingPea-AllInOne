import os
import lmdb
import pickle
import pandas as pd
from tqdm import tqdm
import torch
from torch_geometric.data import Data
import shutil

from protein2graph import protein2graph
from ligand2graph import ligand2graph

def pdbbind2lmdb(pdbbind_dir, csv_file, output_lmdb, mode='update', map_size=1099511627776):
    """
    Process PDBbind data and store in LMDB.
    
    :param pdbbind_dir: Directory containing PDBbind data
    :param csv_file: Path to pdbbind.csv file containing Kd values
    :param output_lmdb: Path to output LMDB file
    :param mode: 'rewrite' to create a new LMDB, 'update' to update existing or create new
    :param map_size: Maximum size database may grow to; default is 1TB
    """
    if mode == 'rewrite' and os.path.exists(output_lmdb):
        shutil.rmtree(output_lmdb)
    
    if mode == 'update' and os.path.exists(output_lmdb):
        try:
            # Test if the existing LMDB can be read
            with lmdb.open(output_lmdb, readonly=True) as env:
                with env.begin() as txn:
                    test_key = list(txn.cursor().iternext(keys=True, values=False))[0]
                    test_value = txn.get(test_key)
                    if test_value is not None:
                        print(f"Existing LMDB at {output_lmdb} is valid. No action needed.")
                        return
        except Exception as e:
            print(f"Existing LMDB at {output_lmdb} is invalid or empty. Recreating...")
            shutil.rmtree(output_lmdb)

    # Read Kd values from CSV
    kd_data = pd.read_csv(csv_file)
    kd_dict = dict(zip(kd_data['pdb_id'], kd_data['kd_value']))

    env = lmdb.open(output_lmdb, map_size=map_size)

    with env.begin(write=True) as txn:
        for pdb_id in tqdm(os.listdir(pdbbind_dir)):
            pdb_dir = os.path.join(pdbbind_dir, pdb_id)
            if not os.path.isdir(pdb_dir):
                continue

            try:
                # Process protein
                protein_file = os.path.join(pdb_dir, f"{pdb_id}_protein.pdb")
                protein_data = protein2graph(protein_file) if os.path.exists(protein_file) else None

                # Process ligand 
                ligand_file = os.path.join(pdb_dir, f"{pdb_id}_ligand.sdf")
                ligand_data = ligand2graph(ligand_file) if os.path.exists(ligand_file) else None
                
                # Get Kd value
                kd_value = kd_dict.get(pdb_id)

                # Create nested dictionary
                pdb_data = {
                    'protein_graph': protein_data,
                    'ligand_graph': ligand_data,
                    'kd_value': kd_value
                }

                # Store in LMDB
                txn.put(pdb_id.encode(), pickle.dumps(pdb_data))
            
            except Exception as e:
                print(f"Error processing {pdb_id}: {str(e)}")

    env.close()
    print(f"Data processing complete. LMDB database saved to {output_lmdb}")

def read_from_lmdb(lmdb_path, pdb_id):
    """
    Read data for a specific PDB ID from LMDB database.
    
    :param lmdb_path: Path to LMDB database
    :param pdb_id: PDB ID to retrieve
    :return: Dictionary containing protein_graph, ligand_graph, and kd_value
    """
    env = lmdb.open(lmdb_path, readonly=True)
    with env.begin() as txn:
        data = txn.get(pdb_id.encode())
        if data is not None:
            return pickle.loads(data)
    env.close()
    return None