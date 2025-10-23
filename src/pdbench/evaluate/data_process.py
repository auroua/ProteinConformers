import os
import pickle
import shutil
import subprocess
from pathlib import Path
import mdtraj

import MDAnalysis as mda
import numpy as np
import torch
import re
from tqdm import tqdm
from bioemu.convert_chemgraph import get_atom37_from_frames, filter_unphysical_traj, _write_pdb
from bioemu.openfold.np.protein import Protein, to_pdb
from src.pdbench.utils.data_utils import read_sequence, extract_tm_score, extract_coordinates_from_pdb


def compute_tmscores(tm_exec_path, ref_pdb_path, pdb_folder):
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
            cmd = [str(tm_exec_path), str(pdb_file), str(ref_pdb_path)]
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
            # if len(results) == 100:
            #     break
            cmd = [str(tm_exec_path), str(pdb_file), str(ref_pdb_path)]
            try:
                output = subprocess.check_output(cmd, text=True)
                tm_score = extract_tm_score(output)
                results[pdb_file.name] = tm_score
            except subprocess.CalledProcessError as e:
                print(f"Failed to run TMscore for {pdb_file.name}: {e}")

        return results


def save_ca_positions_to_pdb(positions, sequence, output_path, chain_id='A'):
    """
    Save C-alpha atom positions to a PDB file.

    Args:
        positions (np.ndarray): shape (N, 3), 3D coordinates of Cα atoms
        sequence (str): amino acid sequence corresponding to the positions
        output_path (str or Path): path to the output PDB file
        chain_id (str): chain identifier
    """


    aa3 = {
        'A': 'ALA', 'C': 'CYS', 'D': 'ASP', 'E': 'GLU', 'F': 'PHE',
        'G': 'GLY', 'H': 'HIS', 'I': 'ILE', 'K': 'LYS', 'L': 'LEU',
        'M': 'MET', 'N': 'ASN', 'P': 'PRO', 'Q': 'GLN', 'R': 'ARG',
        'S': 'SER', 'T': 'THR', 'V': 'VAL', 'W': 'TRP', 'Y': 'TYR'
    }

    with open(output_path, 'w') as f:
        for i, (coord, aa) in enumerate(zip(positions, sequence), start=0):
            resname = aa3.get(aa.upper(), 'UNK')
            x, y, z = coord
            f.write(
                f"ATOM  {i:5d}  CA  {resname} {chain_id}{i:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n"
            )
        f.write("END\n")


def gen_xtc_trajectories():
    pass


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


def get_native_structure():
    pass


def convert_bioemu_npz_to_pdb(data_path: str,
                              pdb_id: str,
                              save_path: str
                              ):
    # TODO: This program has issues and can not be run correctly. The coordinates of the converted is not correct.
    all_pdbs = get_all_decoy_file_path(data_path, pdb_id, False)
    pdb_files = all_pdbs[pdb_id]
    # sequences = [np.load(f)["sequence"].item() for f in pdb_files if "topology" not in f]
    # positions = torch.tensor(np.concatenate([np.load(f)["pos"] for f in pdb_files if "topology" not in f]))
    for pdb_file in tqdm(pdb_files):
        if "topology" in pdb_file:
            continue

        print(pdb_file)
        file = np.load(pdb_file)
        c_alpha_position = file["pos"]
        sequence = file["sequence"].item()

        pdb_id = pdb_file.split('.')[0].split("/")[-1]
        start_idx = int(re.findall(r'\d+', pdb_id)[0])
        for i in range(start_idx, start_idx + len(c_alpha_position)):
            pdb_save_path = os.path.join(save_path, f"{pdb_id}_{i}.pdb")
            save_ca_positions_to_pdb(c_alpha_position[i], sequence, pdb_save_path)


def convert_bioemu_npz_to_pdb_new_bk(topoloty_file_path: str,
                                     pdb_id: str,
                                     save_path: str
                                     ):
    position_list = []
    aa_frame_list = []
    files = [os.path.join(topoloty_file_path, f) for f in os.listdir(topoloty_file_path) if f.endswith(".npz")]
    file_name_list = [file.split("/")[-1].split(".")[0] for file in files]
    for pdb_file_path in files:
        protein = np.load(pdb_file_path)
        if "pos" not in protein:
            print(pdb_file_path)
            continue
        position_list.append(protein["pos"])
        aa_frame_list.append(protein["node_orientations"])

    sequence_path = os.path.join(topoloty_file_path, "sequence.fasta")
    sequence = read_sequence(sequence_path)

    pos = torch.tensor(np.concatenate(position_list), dtype=torch.float32)
    node_orientations = torch.tensor(np.concatenate(aa_frame_list), dtype=torch.float32)

    pos = pos * 10.0  # Convert to Angstroms
    pos = pos - pos.mean(axis=1, keepdims=True)  # Center every structure at the origin

    for idx, file in enumerate(file_name_list):
        # topology_path = os.path.join(save_path, pdb_id, f"{file}.pdb")
        topology_path = os.path.join(save_path, f"{file}.pdb")
        _write_pdb(
            pos=pos[idx],
            node_orientations=node_orientations[idx],
            sequence=sequence,
            filename=topology_path,
        )


def convert_bioemu_npz_to_pdb_new(topoloty_file_path: str,
                                  pdb_id: str,
                                  save_path: str
                                  ):
    position_list = []
    aa_frame_list = []
    files = [os.path.join(topoloty_file_path, f) for f in os.listdir(topoloty_file_path) if f.endswith(".npz")]
    file_name_list = [file.split("/")[-1].split(".")[0] for file in files]
    for pdb_file_path in files:
        protein = np.load(pdb_file_path)
        if "pos" not in protein:
            print(pdb_file_path)
            continue
        position_list.append(protein["pos"])
        aa_frame_list.append(protein["node_orientations"])

    sequence_path = os.path.join(topoloty_file_path, "sequence.fasta")
    sequence = read_sequence(sequence_path)

    pos = torch.tensor(np.concatenate(position_list), dtype=torch.float32)

    for idx, file in tqdm(enumerate(file_name_list)):
        start_idx = int(re.findall(r'\d+', file)[0])

        for nested_idx in range(len(position_list[idx])):
            total_idx = start_idx + nested_idx
            topology_path = os.path.join(save_path, f"{pdb_id}_{total_idx}.pdb")
            temp_pos = torch.tensor(position_list[idx][nested_idx], dtype=torch.float32)
            temp_pos *= 10.0
            temp_pos -= temp_pos.mean(axis=0, keepdim=True)
            temp_node_orientations = torch.tensor(aa_frame_list[idx][nested_idx], dtype=torch.float32)
            _write_pdb(
                pos=temp_pos,
                node_orientations=temp_node_orientations,
                sequence=sequence,
                filename=topology_path,
            )


def calculate_tm_score(native_pdb_path: str,
                       target_pdb_files: str | list[str],
                       tm_exec_path: str):
    results = compute_tmscores(tm_exec_path, native_pdb_path, target_pdb_files)
    # for k, v in results.items():
    #     print(f"{k}: {v}")
    #
    # print(len(results))
    return results


def show_xtc_file(topoloty_file_path: str,
                  xtc_file_path: str
                  ):
    if topoloty_file_path and xtc_file_path:
        u = mda.Universe(topoloty_file_path,
                         xtc_file_path)
    else:
        # Provide both topology and trajectory
        u = mda.Universe("/home/bingwu/Downloads/Dynamics/T1024/topology.pdb", "/home/bingwu/Downloads/Dynamics/T1024/samples.xtc")
    # Access frames
    for ts in u.trajectory:
        print(f"Frame: {ts.frame}, Time: {ts.time} ps")
        print("Positions:\n", u.atoms.positions.shape)


def parse_ca_positions_from_pdb(pdb_path):
    positions = []

    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                positions.append([x, y, z])

    return torch.tensor(positions, dtype=torch.float32)



def convert_bioemu_to_xtc(
        pdb_folder_path: str,
        pdb_id: str,
        sorted_keys: list[str],
        filter_samples: bool = True,
        save_path: str = None
):
    position_list = []
    aa_frame_list = []
    for key in sorted_keys:
        pdb_file_path = os.path.join(pdb_folder_path, f"{key}".replace("pdb", "npz"))
        if not os.path.exists(pdb_file_path):
            continue
        protein = np.load(pdb_file_path)
        if "pos" not in protein:
            print(pdb_file_path)
            continue
        position_list.append(protein["pos"])
        aa_frame_list.append(protein["node_orientations"])

    print(f"The num of osition files are {len(position_list)}, and the num of amino acids is {len(aa_frame_list)}")

    sequence_path = os.path.join(pdb_folder_path, "sequence.fasta")
    sequence = read_sequence(sequence_path)

    pos = torch.tensor(np.concatenate(position_list), dtype=torch.float32)
    node_orientations = torch.tensor(np.concatenate(aa_frame_list), dtype=torch.float32)

    pos = pos * 10.0  # Convert to Angstroms
    pos = pos - pos.mean(axis=1, keepdims=True)  # Center every structure at the origin

    topology_path = os.path.join(save_path, "topology_bioemu.pdb")
    _write_pdb(
        pos=pos[0],
        node_orientations=node_orientations[0],
        sequence=sequence,
        filename=topology_path,
    )

    batch_size, _, _ = pos.shape
    xyz = []
    for i in range(batch_size):
        atom_37, atom_37_mask, _ = get_atom37_from_frames(
            pos=pos[i], node_orientations=node_orientations[i], sequence=sequence
        )
        xyz.append(atom_37.view(-1, 3)[atom_37_mask.flatten()].cpu().numpy())

    topology = mdtraj.load_topology(topology_path)

    # Convert positions back to nm for saving to xtc.
    traj = mdtraj.Trajectory(xyz=np.stack(xyz) * 0.1, topology=topology)

    if filter_samples:
        num_samples_unfiltered = len(traj)
        traj = filter_unphysical_traj(traj)
        print(
            f"Filtered {num_samples_unfiltered} samples down to {len(traj)} "
            "based on structure criteria. Filtering can be disabled with `--filter_samples=False`."
        )

    traj.superpose(reference=traj, frame=0)
    xtc_path = os.path.join(save_path, f"bioemu_{pdb_id}.xtc")
    traj.save_xtc(xtc_path)


def convert_bioemu_npz2pdb():
    bioemu_npz_file_path = "/home/bingwu/Downloads/Dynamics/T1024/batch_0000000_0000001.npz"
    positions = np.load(bioemu_npz_file_path)["pos"]
    sequence = np.load(bioemu_npz_file_path)["sequence"].item()
    node_orientations = np.load(bioemu_npz_file_path)["node_orientations"]
    print(node_orientations.shape)

    pos = positions * 10.0  # Convert to Angstroms
    pos = pos - pos.mean(axis=1, keepdims=True)

    num_residues = pos.shape[1]

    atom_37, atom_37_mask, aatype = get_atom37_from_frames(
        pos=torch.tensor(pos[0]), node_orientations=torch.tensor(node_orientations[0]), sequence=sequence
    )

    protein = Protein(
        atom_positions=atom_37.cpu().numpy(),
        aatype=aatype.cpu().numpy(),
        atom_mask=atom_37_mask.cpu().numpy(),
        residue_index=np.arange(num_residues, dtype=np.int64),
        b_factors=np.zeros((num_residues, 37)),
    )
    with open("test.pdb", "w") as f:
        f.write(to_pdb(protein))


def generate_bioemu_xtc(native_file_path,
                        bioemu_pdbs: str,
                        tm_exec_path: str,
                        bioemu_save_path: str,
                        bioemu_npz_file_path: str):
    results = calculate_tm_score(native_file_path, bioemu_pdbs, tm_exec_path)
    sorted_pdb_files = dict(sorted(results.items(), key=lambda item: item[1], reverse=True))
    sorted_keys = [k for k, v in sorted(results.items(), key=lambda item: item[1], reverse=True)]


    with open(os.path.join(bioemu_save_path, f"bioemu_tm_score.pkl"), "wb") as f:
        pickle.dump(sorted_pdb_files, f)

    # with open("/home/bingwu/Downloads/Dynamics/T1024_xtc/bioemu/T1024/bioemu_tm_score.pkl", "rb") as f:
    #     sorted_pdb_files = pickle.load(f)
    sorted_keys = list(sorted_pdb_files.keys())

    sorted_keys = sorted_keys[::3]

    convert_bioemu_to_xtc(bioemu_npz_file_path,
                          pdb_id="T1024",
                          sorted_keys=sorted_keys,
                          filter_samples=True,
                          save_path=bioemu_save_path
                          )


def generate_esmdiff_xtc(
        native_file_path: str,
        esmdiff_pdb_file_path: str,
        tm_exec_path: str,
        esmdiff_save_path: str,
        pdb_id: str,
        filter_samples: bool = True
):
    results = calculate_tm_score(native_file_path, esmdiff_pdb_file_path, tm_exec_path)
    sorted_pdb_files = dict(sorted(results.items(), key=lambda item: item[1], reverse=True))
    sorted_keys = [k for k, v in sorted(results.items(), key=lambda item: item[1], reverse=True)]

    with open(os.path.join(esmdiff_save_path, f"esmdiff_tm_score.pkl"), "wb") as f:
        pickle.dump(sorted_pdb_files, f)
        pickle.dump(sorted_keys, f)

    topology_path = os.path.join(esmdiff_save_path, "topology_esmdiff.pdb")

    position_list = []
    for idx, key in enumerate(sorted_keys):
        pdb_file_path = os.path.join(esmdiff_pdb_file_path, f"{key}")
        if idx == 0:
            shutil.copy(pdb_file_path, topology_path)
        try:
            position = extract_coordinates_from_pdb(pdb_file_path)
        except Exception as e:
            print(pdb_file_path)
            print(e)
            continue
        position_list.append(position)

    topology = mdtraj.load_topology(topology_path)

    # Convert positions back to nm for saving to xtc.
    traj = mdtraj.Trajectory(xyz=np.stack(position_list) * 0.1, topology=topology)

    if filter_samples:
        num_samples_unfiltered = len(traj)
        traj = filter_unphysical_traj(traj)
        print(
            f"Filtered {num_samples_unfiltered} samples down to {len(traj)} "
            "based on structure criteria. Filtering can be disabled with `--filter_samples=False`."
        )

    traj.superpose(reference=traj, frame=0)
    xtc_path = os.path.join(esmdiff_save_path, f"esmdiff_{pdb_id}.xtc")
    traj.save_xtc(xtc_path)


def generate_gt_xtc(
        native_file_path: str,
        gt_file_path: list[str],
        gt_conformation_mapping: dict[str, str],
        tm_exec_path: str,
        gt_save_path="/home/bingwu/Downloads/Dynamics/T1024_xtc/gt/T1024/",
        pdb_id="T1024",
        filter_samples: bool = True
):
    # results = calculate_tm_score(native_file_path, gt_file_path, tm_exec_path)
    # sorted_pdb_files = dict(sorted(results.items(), key=lambda item: item[1], reverse=True))
    # sorted_keys = [k for k, v in sorted(results.items(), key=lambda item: item[1], reverse=True)]
    #
    # with open(os.path.join(gt_save_path, f"gt_tm_score.pkl"), "wb") as f:
    #     pickle.dump(sorted_pdb_files, f)
    #     pickle.dump(sorted_keys, f)

    with open(os.path.join(gt_save_path, f"gt_tm_score.pkl"), "rb") as f:
        sorted_pdb_files = pickle.load(f)
        sorted_keys = pickle.load(f)

    topology_path = os.path.join(gt_save_path, "topology_gt.pdb")

    position_list = []
    copy_flag = False
    for idx, key in enumerate(sorted_keys):
        pdb_file_path = gt_conformation_mapping[key]
        position = extract_coordinates_from_pdb(pdb_file_path)
        if position.shape[0] != 6490:
            continue
        if not copy_flag:
            shutil.copy(pdb_file_path, topology_path)
            copy_flag = True
        position_list.append(position)

    # position_list = [position for position in position_list if position.shape[0] == 6490][::5]
    position_list = [position for position in position_list if position.shape[0] == 6490]

    topology = mdtraj.load_topology(topology_path)

    # Convert positions back to nm for saving to xtc.
    traj = mdtraj.Trajectory(xyz=np.stack(position_list) * 0.1, topology=topology)

    if filter_samples:
        num_samples_unfiltered = len(traj)
        traj = filter_unphysical_traj(traj)
        print(
            f"Filtered {num_samples_unfiltered} samples down to {len(traj)} "
            "based on structure criteria. Filtering can be disabled with `--filter_samples=False`."
        )

    traj.superpose(reference=traj, frame=0)
    xtc_path = os.path.join(gt_save_path, f"gt_{pdb_id}.xtc")
    traj.save_xtc(xtc_path)


if __name__ == "__main__":
    base_path = "/mnt/rna01/chenw/Datasets/Protein_Dynamic"

    native_file_path = f"{base_path}/T1024_GT/native/T1024.pdb"
    gt_decoy_file_path = f"{base_path}/T1024_GT/decoys_MD/"

    bioemu_decoy_path = f"{base_path}/T1024_bioemu_pdbs"
    esmdiff_decoy_path = f"{base_path}/T1024_step1000_eps1e-05_N3000_20250502-152418"
    bioemu_pdbs = f"{base_path}/T1024_bioemu_pdbs/"
    bioemu_npz_files = f"{base_path}/T1024_bioemu_npz/"
    esmdiff_pdbs = f"{base_path}/T1024_step1000_eps1e-05_N3000_20250502-152418/"
    tm_exec_path = "/home/bingwu/WorkSpaces/ProteinDynamicBenchmark/scripts/TMscore"
    # convert_bioemu_npz_to_pdb("/home/bingwu/Downloads/Dynamics/T1024/",
    #                           "T1024",
    #                           "/home/bingwu/Downloads/Dynamics/T1024_bioemu_pdbs/")

    # step 1. Convert bioemu generated npz files to pdb
    # convert_bioemu_npz_to_pdb_new(
    #     topoloty_file_path="/home/bingwu/Downloads/Dynamics/T1024/",
    #     pdb_id="T1024",
    #     save_path="/home/bingwu/Downloads/Dynamics/T1024_bioemu_pdbs_new",
    # )



    # step 2. calculate tm_scores of generate pdbs with the native pdb file
    # gt_conformations = get_all_decoy_file_path(gt_decoy_file_path, "T1024", gt_decoy_path=True)["T1024"]
    # results = calculate_tm_score(native_file_path, esmdiff_pdbs, tm_exec_path)
    # # results = calculate_tm_score(native_file_path, gt_conformations[:100], tm_exec_path)
    #

    # sorted_pdb_files = dict(sorted(results.items(), key=lambda item: item[1], reverse=True))
    # sorted_keys = [k for k, v in sorted(results.items(), key=lambda item: item[1], reverse=True)]

    # bioemu_save_path = "/home/bingwu/Downloads/Dynamics/T1024_xtc/bioemu/T1024/"

    # generate_bioemu_xtc(
    #     native_file_path=native_file_path,
    #     bioemu_pdbs=bioemu_pdbs,
    #     tm_exec_path=tm_exec_path,
    #     bioemu_save_path=bioemu_npz_files,
    #     bioemu_npz_file_path=bioemu_npz_files
    # )

    # generate_esmdiff_xtc(
    #     native_file_path=native_file_path,
    #     esmdiff_pdb_file_path="/home/bingwu/Downloads/Dynamics/T1024_step1000_eps1e-05_N3000_20250502-152418/",
    #     tm_exec_path=tm_exec_path,
    #     esmdiff_save_path="/home/bingwu/Downloads/Dynamics/T1024_xtc/esmdiff/T1024/",
    #     pdb_id="T1024"
    # )

    gt_conformations = get_all_decoy_file_path(gt_decoy_file_path, "T1024", gt_decoy_path=True)["T1024"]
    gt_conformation_mapping = {key.split("/")[-1][:-4]: key for key in gt_conformations}
    generate_gt_xtc(
        native_file_path=native_file_path,
        gt_file_path=gt_conformations,
        gt_conformation_mapping=gt_conformation_mapping,
        tm_exec_path=tm_exec_path,
        # gt_save_path="/home/bingwu/Downloads/Conformation Generation Project/Dynamics/T1024_xtc/gt/T1024/",
        gt_save_path=f"{base_path}/T1024_xtc/gt_new/T1024",
        pdb_id="T1024",
        filter_samples=True
    )