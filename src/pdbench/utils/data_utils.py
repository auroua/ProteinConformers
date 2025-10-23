from Bio import SeqIO
from biotite.structure import (rmsd, superimpose, AtomArray,
                               AffineTransformation, stack)
# from biotite.structure import tm_score, superimpose_structural_homologs
import numpy as np
import re
import subprocess
from pathlib import Path
import mdtraj
import pickle
import os
import MDAnalysis as mda
from Bio.PDB import PDBParser, PPBuilder, Polypeptide
from Bio.PDB.PDBExceptions import PDBException

from bioemu.convert_chemgraph import filter_unphysical_traj
from src.pdbench.third_party.esm.utils.structure.protein_chain import ProteinChain


def parse_fasta(fasta_file_path: str) -> dict[str, str]:
    """
    This method is used to load fasta file, and return a dict object with keys and sequences.
    :param fasta_file_path: fasta file path.
    :return: a python dict object with sequence keys and sequences.
    """
    fasta_dict: dict[str, str] = {}
    records = SeqIO.parse(fasta_file_path, "fasta")
    for record in records:
        fasta_dict[record.id] = str(record.seq)
    return fasta_dict


def get_pdb_structure(pdb_file_path: str,
                      chain_id: str = 'A',
                      chain_id_filter: bool = True) -> ProteinChain:
    pdb_data = ProteinChain.from_pdb(pdb_file_path, chain_id=chain_id, chain_id_filter=chain_id_filter)
    return pdb_data


def tm_score(**kwargs):
    pass


def calculate_tm_score(reference: AtomArray,
                       subject: AtomArray) -> tuple[float, AffineTransformation]:
    # superimposed, transform, ref_indices, sub_indices = superimpose_structural_homologs(
    #     reference, subject, max_iterations=20
    # )
    superimposed, transform = superimpose(reference, stack([subject]))
    score = tm_score(reference, superimposed[0], np.arange(len(reference)), np.arange(len(reference)),
                     reference_length="reference")
    return score, transform


def calculate_rmsd_score(reference: AtomArray,
                         subject: AtomArray) -> tuple[float, AffineTransformation]:
    superimposed, transform = superimpose(reference, stack([subject]))
    score = rmsd(reference, superimposed)
    return score[0], transform


def get_c_alpha_atom_array(pdb_data: ProteinChain) -> AtomArray:
    atom_array = pdb_data.atom_array
    return atom_array[atom_array.atom_name == 'CA']


def get_c_alpha_idxs(pdb_data: ProteinChain) -> list[int]:
    atom_array = pdb_data.atom_array
    return [i for i in range(len(atom_array.atom_name)) if atom_array.atom_name[i] == "CA"]


def get_native_structure(fasta_file_path: str):
    pdb_ids = []
    with open(fasta_file_path, "r") as f:
        for line in f:
            if ">" in line:
                pdb_id = line.split(' ')[0]
                pdb_ids.append(pdb_id[1:])
    print(pdb_ids)


def extract_coordinates_from_pdb(pdb_file_path: str,
                                 atom_type: str = None):
    coordinates = []

    with open(pdb_file_path, 'r') as f:
        for line in f:
            if line.startswith("ATOM"):
                # Columns 31-54 are x, y, z
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                if atom_type is None:
                    coordinates.append([x, y, z])
                else:
                    atom_name = line[12:16].strip().upper()
                    if atom_name == atom_type:
                        coordinates.append([x, y, z])
    return np.array(coordinates)


def extract_tm_score(output_text):
    match = re.search(r"TM-score\s*=\s*([\d.]+)", output_text)
    if match:
        return float(match.group(1))
    else:
        raise ValueError("TM-score not found in output.")


def compute_zhang_tmscores(tm_exec_path: str,
                           target_pdb_path: str,
                           native_pdb_path: str):
    """
    Compute TM-scores for all PDB files in a folder against a reference.

    Args:
        tm_exec_path (str or Path): path to the TMscore executable
        ref_pdb_path (str or Path): path to the reference PDB file
        pdb_folder (str or Path): folder containing the PDB files to compare

    Returns:
        dict: filename -> TM-score
    """

    cmd = [tm_exec_path, target_pdb_path, native_pdb_path]
    try:
        output = subprocess.check_output(cmd, text=True)
        tm_score = extract_tm_score(output)
    except subprocess.CalledProcessError as e:
        print(f"Failed to run TMscore for {target_pdb_path}: {e}")
        print(e)
        return None
    return tm_score


def compute_zhang_tmscores_folder(tm_exec_path: str,
                                  ref_pdb_path: str,
                                  pdb_folder: list[str] | str):
    """
    Compute TM-scores for all PDB files in a folder against a reference.

    Args:
        tm_exec_path (str or Path): path to the TMscore executable
        ref_pdb_path (str or Path): path to the reference PDB file
        pdb_folder (str or Path): folder containing the PDB files to compare

    Returns:
        dict: filename -> TM-score
    """

    if isinstance(pdb_folder, list):
        results = {}

        for pdb_file in pdb_folder:
            cmd = [tm_exec_path, str(pdb_file), ref_pdb_path]
            try:
                output = subprocess.check_output(cmd, text=True)
                tm_score = extract_tm_score(output)
                results[pdb_file.split('/')[-1][:-4]] = tm_score
            except subprocess.CalledProcessError as e:
                print(f"Failed to run TMscore for {pdb_file.name}: {e}")

        return results
    else:
        pdb_folder = Path(pdb_folder)
        results = {}

        for pdb_file in pdb_folder.glob("*.pdb"):
            cmd = [tm_exec_path, str(pdb_file), ref_pdb_path]
            try:
                output = subprocess.check_output(cmd, text=True)
                tm_score = extract_tm_score(output)
                results[pdb_file.name] = tm_score
            except subprocess.CalledProcessError as e:
                print(f"Failed to run TMscore for {pdb_file.name}: {e}")

        return results


def read_sequence(seq_file_path: str):
    total_lines = ""
    with open(seq_file_path, 'r') as f:
        for idx, line in enumerate(f):
            if idx == 0:
                continue
            total_lines += line.strip()
    return total_lines


def save_xtc_file(topology_file_path: str,
                  positions: list[np.array],
                  save_path: str,
                  pdb_id: str,
                  model_type: str,
                  filter_samples: bool = True,
                  threashod: float = None):
    topology = mdtraj.load_topology(topology_file_path)

    # Convert positions back to nm for saving to xtc.
    traj = mdtraj.Trajectory(xyz=np.stack(positions) * 0.1, topology=topology)

    if filter_samples:
        num_samples_unfiltered = len(traj)
        traj = filter_unphysical_traj(traj)
        print(
            f"Filtered {num_samples_unfiltered} samples down to {len(traj)} "
            "based on structure criteria. Filtering can be disabled with `--filter_samples=False`."
        )

    traj.superpose(reference=traj, frame=0)
    if threashod:
        xtc_path = os.path.join(save_path, f"{model_type}_{pdb_id}_0_5.xtc")
    else:
        xtc_path = os.path.join(save_path, f"{model_type}_{pdb_id}.xtc")
    traj.save_xtc(xtc_path)


def generate_xtc(
        native_file_path: str,
        pdb_file_folder: str | list[str],
        tm_exec_path: str,
        xtc_save_path: str,
        ai_models: str,
):
    results = compute_zhang_tmscores_folder(tm_exec_path, native_file_path, pdb_file_folder)
    sorted_pdb_files = dict(sorted(results.items(), key=lambda item: item[1], reverse=True))
    sorted_keys = [k for k, v in sorted(results.items(), key=lambda item: item[1], reverse=True)]

    with open(os.path.join(xtc_save_path, f"{ai_models}_tm_score.pkl"), "wb") as f:
        pickle.dump(sorted_pdb_files, f)
        pickle.dump(sorted_keys, f)
    return sorted_pdb_files, sorted_keys


def show_xtc_file(topoloty_file_path: str,
                  xtc_file_path: str
                  ):
    u = mda.Universe(topoloty_file_path,
                     xtc_file_path)
    # Access frames
    for ts in u.trajectory:
        print(f"Frame: {ts.frame}, Time: {ts.time} ps")
        print("Positions:\n", u.atoms.positions.shape)


def get_all_decoy_file_path(data_path: str,
                            pdb_id: str,
                            gt_decoy_path: bool = True):
    if gt_decoy_path:
        pdb_files = []
        folders = os.listdir(data_path)
        for folder in folders:
            folder_path = os.path.join(data_path, folder)
            folder_files = os.listdir(folder_path)
            folder_pdb_files = [os.path.join(folder_path, f) for f in folder_files if f.endswith(".pdb")]
            pdb_files += folder_pdb_files
    else:
        temp_pdb_files = os.listdir(data_path)
        pdb_files = [os.path.join(data_path, f) for f in temp_pdb_files if f.endswith("pdb") or f.endswith("npz")]
    return {pdb_id: pdb_files}


def extract_pdb_sequences_new(pdb_file, use_seqres=False):
    """
    从 PDB 文件中提取各条链的序列。

    参数：
        pdb_file (str or file-like): PDB 文件路径或类文件对象
        use_seqres (bool):
            False（默认）—基于 ATOM 记录提取序列，序号跳跃处自动补 'X'；
            True —直接读取 SEQRES 头部声明的完整序列。

    返回：
        dict: {chain_id (str) → sequence (str)}
    """
    mode = 'pdb-seqres' if use_seqres else 'pdb-atom'
    seqs = {}
    for rec in SeqIO.parse(pdb_file, mode):
        cid = rec.annotations.get('chain', '?')
        seqs[cid] = str(rec.seq)
    return seqs


def extract_sequence_from_pdb(pdb_file):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("structure", pdb_file)

    ppb = PPBuilder()
    # ppb = PPBuilderLocal()
    for model in structure:
        for chain in model:
            sequence = ''
            for pp in ppb.build_peptides(chain):
                sequence += pp.get_sequence()
            # print(f"Chain {chain.id}: {sequence}")
    return sequence


