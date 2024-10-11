import os
import argparse
import logging
import pickle
from multiprocessing import Pool
import pandas as pd
from tqdm import tqdm
import time
import tempfile

from rdkit import Chem
from ligand2graph import ligand2graph

def initialize_logger(log_file='preprocess.log'):
    """
    Initialize the logger.
    """
    logging.basicConfig(
        filename=log_file,
        filemode='a',
        format='%(asctime)s [%(levelname)s] %(message)s',
        level=logging.INFO
    )

def load_labels(csv_file):
    """
    Load labels from a CSV file.

    Args:
        csv_file (str): Path to the CSV file.

    Returns:
        dict: A dictionary with index as keys and labels as values.
    """
    df = pd.read_csv(csv_file)
    if 'label' not in df.columns:
        logging.error("CSV file must contain a 'label' column.")
        raise ValueError("Invalid CSV file format.")
    labels = {idx: label for idx, label in enumerate(df['label'])}
    return labels

def process_sdf(args):
    """
    Process a single SDF file and convert it to a PyG graph structure.

    Args:
        args (tuple): A tuple containing (index, sdf_path, label).

    Returns:
        tuple: (index, {'conformer': [list of graph structures], 'label': float})
    """
    index, sdf_path, label = args
    try:
        conformer_graphs = []

        # Load the molecule(s) from the SDF file
        suppl = Chem.SDMolSupplier(sdf_path, removeHs=False)
        mols = [mol for mol in suppl if mol is not None]

        if not mols:
            logging.error(f"No valid molecules found in SDF file at index {index}.")
            return (index, {'conformer': [], 'label': label})

        for mol in mols:
            # Remove hydrogen atoms
            mol_no_h = Chem.RemoveHs(mol)

            # Write mol_no_h to a temporary SDF file
            with tempfile.NamedTemporaryFile(suffix='.sdf') as temp_sdf:
                writer = Chem.SDWriter(temp_sdf.name)
                writer.write(mol_no_h)
                writer.close()

                # Process the molecule to get the graph structure
                try:
                    data = ligand2graph(temp_sdf.name)
                except Exception as e_ligand2graph:
                    logging.error(f"ligand2graph failed at index {index}, error: {e_ligand2graph}")
                    continue  # Skip to the next molecule

                if isinstance(data, list):
                    conformer_graphs.extend(data)
                else:
                    conformer_graphs.append(data)

        if not conformer_graphs:
            # Record empty list index to lig2graph_failed.txt
            with open('lig2graph_failed.txt', 'a') as f_failed:
                f_failed.write(f"{index}\n")
            logging.warning(f"No conformers generated at index {index}. Recorded to lig2graph_failed.txt.")

        return (index, {'conformer': conformer_graphs, 'label': label})
    except Exception as e:
        logging.error(f"Failed to process SDF file at index {index}, file: {sdf_path}, error: {e}")
        return (index, {'conformer': [], 'label': label})

def preprocess(conformer_dir, csv_file, output_pkl, num_processes):
    """
    Main preprocessing function that processes all SDF files and saves them as a serialized pickle file.

    Args:
        conformer_dir (str): Directory containing the SDF files.
        csv_file (str): Path to the CSV file containing labels.
        output_pkl (str): Path to the output pickle file.
        num_processes (int): Number of processes for parallel processing.
    """
    labels = load_labels(csv_file)
    tasks = []
    for index in tqdm(labels.keys(), desc="Preparing tasks"):
        sdf_path = os.path.join(conformer_dir, f"{index}.sdf")
        if os.path.exists(sdf_path):
            tasks.append((index, sdf_path, labels[index]))
        else:
            logging.warning(f"SDF file does not exist at index {index}, path: {sdf_path}")

    data_dict = {}
    with Pool(processes=num_processes) as pool:
        for result in tqdm(pool.imap_unordered(process_sdf, tasks), total=len(tasks), desc="Processing SDF files"):
            index, data = result
            if data['conformer']:
                data_dict[index] = data
            else:
                # If no valid conformers, also record to lig2graph_failed.txt
                with open('lig2graph_failed.txt', 'a') as f_failed:
                    f_failed.write(f"{index}\n")
                logging.warning(f"No valid conformers generated at index {index}. Recorded.")
                data_dict[index] = {'conformer':[],'label':labels[index]}

    # Save as a pickle file
    with open(output_pkl, 'wb') as f:
        pickle.dump(data_dict, f)
    logging.info(f"Preprocessing completed. Data saved to {output_pkl}")

def main():
    parser = argparse.ArgumentParser(description='Preprocessing script to convert SDF files to graph structures and serialize them.')
    parser.add_argument('--conformer_dir', type=str, default='../conformer', help='Directory containing SDF files.')
    parser.add_argument('--csv_file', type=str, default='../raw/train.csv', help='Path to the CSV file containing labels.')
    parser.add_argument('--output_pkl', type=str, default='../processed/data_nonmeta.pkl', help='Path to the output pickle file.')
    parser.add_argument('--log_file', type=str, default='preprocess.log', help='Name of the log file.')
    parser.add_argument('--num_processes', type=int, default=48, help='Number of processes for parallel processing.')
    args = parser.parse_args()

    initialize_logger(args.log_file)
    logging.info("Preprocessing script started.")
    start_time = time.time()

    preprocess(args.conformer_dir, args.csv_file, args.output_pkl, args.num_processes)

    end_time = time.time()
    total_time = end_time - start_time
    logging.info(f"Preprocessing script completed in {total_time:.2f} seconds.")

if __name__ == "__main__":
    main()