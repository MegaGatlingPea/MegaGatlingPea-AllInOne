import os
import lmdb
import pickle
import pandas as pd
from tqdm import tqdm
import torch
from torch_geometric.data import Data
import shutil

from pocket2graph import pocket2graph
from ligand2graph import ligand2graph

def pdbbind2lmdb(pdbbind_dir='./../../Testset', csv_file='./../pdbbind.csv', output_lmdb='./../pdbbind.lmdb', mode='rewrite', map_size=1099511627776, batch_size=32):
    """
    Process PDBbind data and store in LMDB with cluster organization using batch writing.
    
    :param pdbbind_dir: Directory containing PDBbind data
    :param csv_file: Path to pdbbind.csv file containing Kd values and cluster information
    :param output_lmdb: Path to output LMDB file
    :param mode: 'rewrite' to create a new LMDB, 'update' to update existing or create new
    :param map_size: Maximum size database may grow to; default is 1TB
    :param batch_size: Number of clusters to write in each batch
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

    # Read CSV file, including Kd values and cluster information
    data = pd.read_csv(csv_file)
    data_dict = data.set_index('pdb_id').to_dict('index')

    env = lmdb.open(output_lmdb, map_size=map_size)
    
    cluster_data = {}
    clusters_processed = 0

    with env.begin(write=True) as txn:
        for pdb_id in tqdm(os.listdir(pdbbind_dir), desc="Processing PDB IDs"):
            pdb_dir = os.path.join(pdbbind_dir, pdb_id)
            if not os.path.isdir(pdb_dir):
                continue

            try:
                # Process protein
                protein_file = os.path.join(pdb_dir, f"{pdb_id}_protein.pdb")
                protein_data = pocket2graph(protein_file) if os.path.exists(protein_file) else None

                # Process ligand
                ligand_file = os.path.join(pdb_dir, f"{pdb_id}_ligand.sdf")
                ligand_data = ligand2graph(ligand_file) if os.path.exists(ligand_file) else None
                
                # Get Kd value and cluster ID
                pdb_info = data_dict.get(pdb_id, {})
                kd_value = pdb_info.get('kd_value')
                cluster_id = pdb_info.get('cluster')

                if cluster_id is not None:
                    cluster_key = f'cluster_{cluster_id}'
                    
                    # Create nested dictionary
                    pdb_data = {
                        'protein_graph': protein_data,
                        'ligand_graph': ligand_data,
                        'kd_value': kd_value
                    }

                    if cluster_key not in cluster_data:
                        cluster_data[cluster_key] = {}
                    cluster_data[cluster_key][pdb_id] = pdb_data

                    # determine if the batch size is reached
                    if len(cluster_data) >= batch_size:
                        for key, cluster_dict in cluster_data.items():
                            txn.put(key.encode(), pickle.dumps(cluster_dict))
                        clusters_processed += len(cluster_data)
                        print(f"Have processed {clusters_processed} clusters")
                        cluster_data.clear()  # clear the current batch data

            except Exception as e:
                print(f"Error processing {pdb_id}: {str(e)}")

        # write the remaining data
        if cluster_data:
            for key, cluster_dict in cluster_data.items():
                txn.put(key.encode(), pickle.dumps(cluster_dict))
            clusters_processed += len(cluster_data)
            print(f"Have processed last {len(cluster_data)} clusters")
            cluster_data.clear()

    env.close()
    print(f"Data processing complete. LMDB database saved to {output_lmdb}")

def read_from_lmdb(lmdb_path='./../pdbbind.lmdb', cluster_id=1, pdb_id=None):
    """
    Read data for a specific cluster or PDB ID from LMDB database.
    
    :param lmdb_path: Path to LMDB database
    :param cluster_id: Cluster ID to retrieve
    :param pdb_id: PDB ID to retrieve (optional)
    :return: Dictionary containing cluster data or specific PDB ID data
    """
    env = lmdb.open(lmdb_path, readonly=True)
    with env.begin() as txn:
        cluster_key = f'cluster_{cluster_id}'.encode()
        cluster_data = txn.get(cluster_key)
        if cluster_data is not None:
            cluster_dict = pickle.loads(cluster_data)
            if pdb_id:
                return cluster_dict.get(pdb_id)
            return cluster_dict
    env.close()
    return None

# Usage example
if __name__ == "__main__":
    pdbbind_dir = "./../../Testset"
    csv_file = "./../pdbbind.csv"
    output_lmdb = "./../pdbbind.lmdb"
    
    # Process and store data
    pdbbind2lmdb(pdbbind_dir, csv_file, output_lmdb, mode='rewrite', batch_size=32)
    
    # Example of reading data
    cluster_id = 1  # Assuming we want to read data from cluster_1
    pdb_id = "4ytc"  # Assuming this PDB ID is in cluster_1
    
    # Read data for the entire cluster
    cluster_data = read_from_lmdb(output_lmdb, cluster_id)
    if cluster_data is not None:
        print(f"Cluster {cluster_id} contains {len(cluster_data)} PDB entries")
    
    # Read data for a specific PDB ID
    pdb_data = read_from_lmdb(output_lmdb, cluster_id, pdb_id)
    if pdb_data is not None:
        print(f"Data for {pdb_id} in cluster {cluster_id}:")
        if pdb_data['protein_graph'] is not None:
            print(f"Protein graph nodes: {pdb_data['protein_graph'].x.shape[0]}")
        if pdb_data['ligand_graph'] is not None:
            print(f"Ligand graph nodes: {pdb_data['ligand_graph'].x.shape[0]}")
        print(f"Kd value: {pdb_data['kd_value']}")
    else:
        print(f"No data found for {pdb_id} in cluster {cluster_id}")