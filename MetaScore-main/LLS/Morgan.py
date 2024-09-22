import os
import numpy as np
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from tqdm import tqdm
from rdkit import RDLogger

# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.*')

data_path = "/mnt/data/pdbbind2020-PL"

with open(f'{data_path}/pdbbind_ids.txt', 'r') as f:
    pdb_ids = [line.strip() for line in f]

fingerprints = []
failed_ids = []

# Create MorganGenerator once
morgan_gen = GetMorganGenerator(radius=3, fpSize=2048)

print("Generating fingerprints...")
for pdb_id in tqdm(pdb_ids):
    ligand_file = os.path.join(data_path, pdb_id, f"{pdb_id}_ligand.sdf")
    try:
        # Read the molecule without any checks
        with Chem.SDMolSupplier(ligand_file, sanitize=False, strictParsing=False) as suppl:
            mol = next(suppl)
        
        if mol is not None:
            # Generate fingerprint without sanitization
            fp = morgan_gen.GetFingerprint(mol)
            fingerprints.append(fp)
        else:
            print(f"Warning: Could not read molecule for {pdb_id}")
            failed_ids.append(pdb_id)

    except Exception as e:
        print(f"Error processing {pdb_id}: {str(e)}")
        failed_ids.append(pdb_id)

# Remove failed PDB IDs
for failed_id in failed_ids:
    pdb_ids.remove(failed_id)

print("Calculating similarity matrix...")
n = len(fingerprints)
similarity_matrix = np.zeros((n, n))

for i in tqdm(range(n)):
    for j in range(i, n):
        similarity = DataStructs.TanimotoSimilarity(fingerprints[i], fingerprints[j])
        similarity_matrix[i, j] = similarity
        similarity_matrix[j, i] = similarity

print("Saving results...")
np.save('morgan_similarity_matrix.npy', similarity_matrix)
with open('pdbbind_ids.txt', 'w') as f:
    for pdb_id in pdb_ids:
        f.write(f"{pdb_id}\n")

print(f"Processed {n} ligands. Failed for {len(failed_ids)} ligands.")
print("Results saved as 'morgan_similarity_matrix.npy' and 'pdbbind_ids.txt'")
