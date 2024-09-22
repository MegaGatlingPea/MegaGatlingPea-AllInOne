import oddt
import numpy as np
from oddt.toolkits import ob
from oddt.fingerprints import SPLIF
from typing import Tuple, List, Dict
import os
import random
from tqdm import tqdm
import multiprocessing as mp

def splif_to_bitvec_and_weights(splif_result, size = 4096):
    bitvec = np.zeros(size, dtype=bool)
    weights = np.zeros(size, dtype=int)
    
    unique_hashes, counts = np.unique(splif_result['hash'], return_counts=True)
    bitvec[unique_hashes] = True
    weights[unique_hashes] = counts
    
    return bitvec, weights

def weighted_tanimoto_similarity(splif1, splif2, size: int = 4096):
    bitvec1, weights1 = splif_to_bitvec_and_weights(splif1, size)
    bitvec2, weights2 = splif_to_bitvec_and_weights(splif2, size)
    
    common_bits = bitvec1 & bitvec2
    weighted_intersection = np.sum(np.minimum(weights1[common_bits], weights2[common_bits]))
    weighted_union = np.sum(np.maximum(weights1, weights2))
    
    return weighted_intersection / weighted_union if weighted_union > 0 else 0.0

def load_molecule(file_path, file_type):
    molecule = next(ob.readfile(file_type, file_path))
    if file_type == 'pdb':
        molecule.protein = True
    return molecule

def calculate_splif(ligand, protein):
    return SPLIF(ligand, protein)

def process_complex(pdb_path, sdf_path):
    protein = load_molecule(pdb_path, 'pdb')
    ligand = load_molecule(sdf_path, 'sdf')
    return calculate_splif(ligand, protein)

def get_pdb_ids(base_path, n = None):
    all_pdb_ids = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    if n is None or n >= len(all_pdb_ids):
        return all_pdb_ids
    return random.sample(all_pdb_ids, n)

def process_complex_wrapper(args):
    base_path, pdb_id = args
    pdb_path = os.path.join(base_path, pdb_id, f"{pdb_id}_protein.pdb")
    sdf_path = os.path.join(base_path, pdb_id, f"{pdb_id}_ligand.sdf")
    return pdb_id, process_complex(pdb_path, sdf_path)

def calculate_similarity_matrix(base_path, pdb_ids):
    n = len(pdb_ids)
    similarity_matrix = np.zeros((n, n))

    with mp.Pool(16) as pool:
        results = list(tqdm(pool.imap(process_complex_wrapper, [(base_path, pdb_id) for pdb_id in pdb_ids]), 
                            total=len(pdb_ids), desc="Processing complexes"))
    
    splif_fingerprints = dict(results)

    for i in tqdm(range(n), desc="Calculating similarities"):
        for j in range(i, n):
            similarity = weighted_tanimoto_similarity(
                splif_fingerprints[pdb_ids[i]], 
                splif_fingerprints[pdb_ids[j]]
            )
            similarity_matrix[i, j] = similarity_matrix[j, i] = similarity

    return similarity_matrix

def main():
    base_path = '/mnt/data/pdbbind2020-PL'
    n_complexes = None
    
    pdb_ids = get_pdb_ids(base_path, n_complexes)
    
    print(f"Number of complexes selected: {len(pdb_ids)}")
    
    similarity_matrix = calculate_similarity_matrix(base_path, pdb_ids)
    
    print("Similarity matrix shape:", similarity_matrix.shape)
    print("Sample of similarity matrix:")
    print(similarity_matrix[:5, :5])

    np.save(f"splif_similarity_matrix_{len(pdb_ids)}.npy", similarity_matrix)
    with open(f"pdb_ids_{len(pdb_ids)}.txt", "w") as f:
        for pdb_id in pdb_ids:
            f.write(f"{pdb_id}\n")

    print(f"Results saved to 'splif_similarity_matrix_{len(pdb_ids)}.npy' and 'pdb_ids_{len(pdb_ids)}.txt'")

if __name__ == "__main__":
    main()