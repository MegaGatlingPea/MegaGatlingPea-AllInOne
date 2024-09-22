import numpy as np
from tmtools import tm_align
from tmtools.io import get_structure, get_residue_data
from tqdm import tqdm
from multiprocessing import Pool
import os

# PDBBind Set
data_path = '/mnt/data/pdbbind2020-PL'

# PDB IDs
with open(f'{data_path}/pdbbind_ids.txt', 'r') as f:
    pdb_ids = [line.strip() for line in f]

num_proteins = len(pdb_ids)

# calculate coordinates and sequences
def load_protein(prot):
    path = f'{data_path}/{prot}/{prot}_protein.pdb'
    s = get_structure(path)
    chain = next(s.get_chains())
    coords, seq = get_residue_data(chain)
    return coords, seq

with Pool(processes=8) as pool:
    protein_data = list(tqdm(pool.imap(load_protein, pdb_ids), total=num_proteins, desc="Loading protein structures"))

coords_list, seq_list = zip(*protein_data)

def tm_score_no_avg(coords1, coords2, seq1, seq2):
    res = tm_align(coords1, coords2, seq1, seq2)
    return res.tm_norm_chain1 

# TM-scores
def calculate_row(i):
    row = np.array([tm_score_no_avg(coords_list[i], coords_list[j], seq_list[i], seq_list[j]) 
                    for j in range(num_proteins)])
    return row

# multiprocessing TM-scores
with Pool(processes=8) as pool:
    tm_matrix = np.array(list(tqdm(pool.imap(calculate_row, range(num_proteins)), 
                                   total=num_proteins, desc="Calculating TM-scores")))

# symmetrise the matrix
tm_matrix_symmetric = (tm_matrix + tm_matrix.T) / 2

# save the result
np.save('tmscore_similarity_matrix.npy', tm_matrix_symmetric)
