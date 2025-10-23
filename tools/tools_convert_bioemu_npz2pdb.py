import shutil
import os
from tqdm import tqdm
import argparse

from src.pdbench.evaluate.data_process import convert_bioemu_npz_to_pdb_new


def show_protein_counts(folder_base_path: str):
    finished_sequences = []
    for folder in os.listdir(folder_base_path):
        folder_path = os.path.join(folder_base_path, folder)
        print(folder, len(os.listdir(folder_path)))
        finished_sequences.append(folder_path)
    return finished_sequences


def rename_pdb_files(pdb_file_path: str):
    for pdb_id in os.listdir(pdb_file_path):
        pdb_path = os.path.join(pdb_file_path, pdb_id)
        if not os.path.isdir(pdb_path):
            continue
        for pdb_file in os.listdir(pdb_path):
            try:
                base_name = pdb_file.split(".")[0]
                pdb_file_id, pdb_seq_id = base_name.split("_")
                new_name = f"{pdb_id}_decoy_{pdb_seq_id}.pdb"
                pdb_target_path = os.path.join(pdb_path, new_name)
                src_pdb_path = os.path.join(pdb_path, pdb_file)
                shutil.move(src_pdb_path, pdb_target_path)
            except ValueError:
                print(f"Skipping invalid filename: {pdb_file}")


def process_npz(args):
    folder_base_path, pdb_save_path, pdb_id = args
    npz_path = os.path.join(folder_base_path, pdb_id)
    target_save_folder_path = os.path.join(pdb_save_path, pdb_id)

    if not os.path.exists(target_save_folder_path):
        os.makedirs(target_save_folder_path)
    else:
        if len(os.listdir(target_save_folder_path)) >= 3000:
            return  # skip if already has many files

    convert_bioemu_npz_to_pdb_new(npz_path, pdb_id, target_save_folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder_base_path", type=str)
    parser.add_argument("--pdb_save_path", type=str)
    args = parser.parse_args()

    finished_sequences = show_protein_counts(args.folder_base_path)

    for source_folder_path in tqdm(finished_sequences):
        print(source_folder_path)

        pdb_id = source_folder_path.split("/")[-1]

        target_save_folder_path = os.path.join(args.pdb_save_path, f"{pdb_id}")
        if not os.path.exists(target_save_folder_path):
            os.makedirs(target_save_folder_path)
        else:
            if len(target_save_folder_path) >= 3000:
                continue
        npz_path = os.path.join(args.folder_base_path, pdb_id)
        convert_bioemu_npz_to_pdb_new(npz_path, pdb_id, target_save_folder_path)