import os
import lmdb
import pickle
import pandas as pd
from tqdm import tqdm
import shutil
import gc
import psutil
import time

from pocket2graph import pocket2graph
from ligand2graph import ligand2graph

def log_memory_usage(stage):
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 ** 3)  # in GB
    print(f"[Memory] {stage}: {mem:.2f} GB")

def pdbbind2lmdb(pdbbind_dir, csv_file, output_dir, mode='rewrite', map_size=1e9, max_pdbs_per_txn=100):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory at {output_dir}")

    # Read CSV and group PDB IDs by cluster
    data = pd.read_csv(csv_file)
    cluster_to_pdb_ids = data.groupby('cluster')['pdb_id'].apply(list).to_dict()
    data_dict = data.set_index('pdb_id').to_dict('index')

    for cluster_id, pdb_ids in tqdm(cluster_to_pdb_ids.items(), desc="Processing Clusters"):
        lmdb_path = os.path.join(output_dir, f"cluster_{cluster_id}.lmdb")

        if mode == 'rewrite' and os.path.exists(lmdb_path):
            shutil.rmtree(lmdb_path)
            print(f"Removed existing LMDB at {lmdb_path}")

        env = lmdb.open(lmdb_path, map_size=map_size)
        txn = env.begin(write=True)
        pdbs_in_txn = 0  # number of PDBs processed in the current transaction

        for pdb_id in tqdm(pdb_ids, desc=f"Processing PDB IDs in cluster {cluster_id}"):
            pdb_dir = os.path.join(pdbbind_dir, pdb_id)
            if not os.path.isdir(pdb_dir):
                continue

            try:
                # Process protein
                protein_file = os.path.join(pdb_dir, f"{pdb_id}_protein.pdb")
                if os.path.exists(protein_file):
                    protein_data = pocket2graph(protein_file)
                else:
                    protein_data = None
                    print(f"Protein file not found for {pdb_id}")

                # Process ligand
                ligand_file = os.path.join(pdb_dir, f"{pdb_id}_ligand.sdf")
                if os.path.exists(ligand_file):
                    ligand_data = ligand2graph(ligand_file)
                else:
                    ligand_data = None
                    print(f"Ligand file not found for {pdb_id}")

                # Get Kd value
                pdb_info = data_dict.get(pdb_id, {})
                kd_value = pdb_info.get('kd_value')

                # Collect pdb_data
                pdb_data = {
                    'protein_graph': protein_data,
                    'ligand_graph': ligand_data,
                    'kd_value': kd_value
                }

                # Store pdb_data in LMDB
                txn.put(pdb_id.encode(), pickle.dumps(pdb_data))
                pdbs_in_txn += 1

                # Clean up
                del protein_data, ligand_data, pdb_data

                # judge whether to commit
                if pdbs_in_txn >= max_pdbs_per_txn:
                    txn.commit()
                    txn = env.begin(write=True)
                    pdbs_in_txn = 0  # reset
                    gc.collect()

            except Exception as e:
                print(f"Error processing {pdb_id}: {str(e)}")

        # commit the remaining transaction
        txn.commit()
        env.close()
        print(f"Finished processing cluster {cluster_id}. LMDB saved at {lmdb_path}")

    print("Data processing complete. All LMDB databases are saved.")


# Usage example
if __name__ == "__main__":
    pdbbind_dir = "/home/megagatlingpea/Data/pdbbind2020-PL"
    csv_file = "./../pdbbind_general.csv"
    output_dir = "./../cluster_pdbbind"

    # Process and store data
    pdbbind2lmdb(pdbbind_dir, csv_file, output_dir, mode='update')
