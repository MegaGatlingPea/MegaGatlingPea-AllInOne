import os
import argparse
import logging
import subprocess
import pandas as pd
import time
from multiprocessing import Pool

def initialize_logger(log_file='conformer_generation.log'):
    """
    Initialize the logger, outputting to both file and console.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # 清除之前的处理器（如果有）
    if logger.hasHandlers():
        logger.handlers.clear()

    # File handler
    fh = logging.FileHandler(log_file, mode='a')
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

def generate_initial_sdf(smiles, output_path):
    """
    Generate initial 3D structure from SMILES and save as SDF.

    Parameters:
        smiles (str): SMILES string.
        output_path (str): Path to save the initial SDF file.

    Returns:
        int: 0 if successful, non-zero otherwise.
    """
    try:
        cmd = [
            'obabel',
            '-:' + smiles,
            '-O', output_path,
            '--gen3D'
        ]
        logging.debug(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                logging.info(f"Initial SDF generated for SMILES.")
                return 0
            else:
                logging.error(f"Failed to generate initial SDF for SMILES: {smiles}.")
                return -1
        else:
            logging.error(f"Error generating initial SDF for SMILES {smiles}: {result.stderr}")
            return result.returncode
    except Exception as e:
        logging.exception(f"Exception during initial SDF generation for SMILES {smiles}: {e}")
        return -1

def generate_conformers_with_confab(input_sdf, output_sdf, num_conformers=100000):
    """
    Use Open Babel Confab to generate multiple conformers from initial SDF.

    Parameters:
        input_sdf (str): Path to the input SDF file.
        output_sdf (str): Path to save the conformers SDF file.
        num_conformers (int): Number of conformers to generate.

    Returns:
        int: 0 if successful, non-zero otherwise.
    """
    try:
        cmd = [
            'obabel',
            input_sdf,
            '-O', output_sdf,
            '--confab',
            '--conf', str(num_conformers)
        ]
        logging.debug(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            if os.path.exists(output_sdf) and os.path.getsize(output_sdf) > 0:
                logging.info(f"Conformers generated with Confab for {input_sdf}.")
                return 0
            else:
                logging.error(f"Failed to generate conformers with Confab for {input_sdf}.")
                return -1
        else:
            logging.error(f"Error generating conformers with Confab for {input_sdf}: {result.stderr}")
            return result.returncode
    except Exception as e:
        logging.exception(f"Exception during conformer generation with Confab for {input_sdf}: {e}")
        return -1

def process_smiles(args):
    """
    Process a single SMILES string to generate and save conformers.

    Parameters:
        args (tuple): Contains index, smiles, num_conformers, output_dir, log_file.
    """
    index, smiles, num_conformers, output_dir, log_file = args

    # 为每个进程初始化独立的日志记录器
    logger = logging.getLogger(f"Process-{index}")
    logger.setLevel(logging.DEBUG)

    # 清除之前的处理器（如果有）
    if logger.hasHandlers():
        logger.handlers.clear()

    # File handler
    fh = logging.FileHandler(log_file, mode='a')
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] [Process-{0}] %(message)s'.format(index))
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    temp_sdf = os.path.join(output_dir, f"temp_{index}.sdf")
    conformer_sdf = os.path.join(output_dir, f"{index}.sdf")

    # Step 1: Generate initial SDF
    result_initial = generate_initial_sdf(smiles, temp_sdf)
    if result_initial != 0:
        logger.error(f"Failed to generate initial SDF for SMILES index {index}.")
        with open('failed_molecules.txt', 'a') as f:
            f.write(f"{index},{smiles},Initial SDF generation failed\n")
        return

    # Step 2: Generate conformers with Confab
    result_confab = generate_conformers_with_confab(temp_sdf, conformer_sdf, num_conformers)
    if result_confab == 0:
        logger.info(f"Successfully generated conformers for SMILES index {index}. Saved to {conformer_sdf}.")
    else:
        logger.error(f"Failed to generate conformers for SMILES index {index}.")
        with open('failed_molecules.txt', 'a') as f:
            f.write(f"{index},{smiles},Conformer generation failed\n")

    # Clean up temporary files
    if os.path.exists(temp_sdf):
        os.remove(temp_sdf)

def main():
    parser = argparse.ArgumentParser(description='Generate and optimize conformers using Open Babel.')
    parser.add_argument('--input_csv', type=str, default='./../raw/train.csv', help='Path to the input CSV file.')
    parser.add_argument('--output_dir', type=str, default='./../conformers', help='Directory to save conformers.')
    parser.add_argument('--num_conformers', type=int, default=100000, help='Number of conformers to generate per molecule.')
    parser.add_argument('--log_file', type=str, default='smiles2sdf.log', help='Log file name.')
    parser.add_argument('--num_processes', type=int, default=1, help='Number of parallel processes.')
    args = parser.parse_args()

    initialize_logger(args.log_file)

    start_time = time.time()
    logging.info("Conformer generation started.")

    try:
        df = pd.read_csv(args.input_csv)
    except Exception as e:
        logging.error(f"Failed to read input CSV file {args.input_csv}. Error: {e}")
        return

    if 'smiles' not in df.columns:
        logging.error("Input CSV file does not contain 'smiles' column.")
        return

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # 创建任务列表，每个任务是一个参数元组
    tasks = [
        (index, row['smiles'], args.num_conformers, args.output_dir, args.log_file)
        for index, row in df.iterrows()
    ]

    # 使用多进程池来并行处理 SMILES
    with Pool(processes=args.num_processes) as pool:
        pool.map(process_smiles, tasks)

    end_time = time.time()
    total_time = end_time - start_time
    logging.info(f"Conformer generation completed in {total_time:.2f} seconds.")

if __name__ == "__main__":
    main()
