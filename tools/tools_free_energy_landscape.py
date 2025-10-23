import os
import pickle
import shutil
import argparse
from tqdm import tqdm
import mdtraj
import numpy as np
import MDAnalysis as mda
import pandas as pd
from MDAnalysis.analysis import align
from scipy.interpolate import griddata
import matplotlib.pyplot as plt

from bioemu.convert_chemgraph import filter_unphysical_traj
from src.pdbench.utils.data_utils import (
    extract_sequence_from_pdb,
    extract_coordinates_from_pdb,
    save_xtc_file,
    get_all_decoy_file_path,
    compute_zhang_tmscores_folder,
)
from src.pdbench.utils.gen_xtc_fel import get_pca_components


def compute_fel(cvs, bins=64, kT=2.494, min_val: float=None):
    """
    Compute Free Energy Landscape (FEL) from Collective Variables (CVs)

    Parameters:
    -----------
    cvs : numpy.ndarray
        2D array of collective variables, shape (n_samples, 2)
    bins : int, optional (default=50)
        Number of bins for 2D histogram
    kT : float, optional (default=2.494)
        Thermal energy in kJ/mol at 298 K

    Returns:
    --------
    X : numpy.ndarray
        X-coordinates for plotting
    Y : numpy.ndarray
        Y-coordinates for plotting
    pmf : numpy.ndarray
        Potential of Mean Force (Free Energy Landscape)
    """
    # Compute 2D histogram with probability density
    hist, xedges, yedges = np.histogram2d(cvs[:, 0], cvs[:, 1], bins=bins, density=True)

    # Avoid log(0) by replacing zero values with a small number
    hist = hist.copy()
    if min_val:
        hist[hist == 0] = min_val * 1e-10
    else:
        hist[hist == 0] = np.min(hist[hist > 0]) * 1e-10
    min_val = np.min(hist[hist > 0])
    # Convert to free energy (PMF)
    # Use negative log probability and scale by thermal energy
    pmf = -kT * np.log(hist.T)

    # Normalize by setting lowest energy to 0
    pmf = pmf - np.min(pmf)

    # Create grid for plotting
    x = 0.5 * (xedges[1:] + xedges[:-1])
    y = 0.5 * (yedges[1:] + yedges[:-1])
    X, Y = np.meshgrid(x, y)
    return X, Y, pmf, min_val


def parse_file_tm_scores(file_path: str):
    """
    Parse a 'decoy,*,*,tmscore,*,*' CSV-like file into {decoy_basename: tm_score}.

    Notes
    -----
    - Keeps only 'decoy' filename stem (without extension) as key.
    - Skips header if present.
    """
    data = {}
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():  # Skip empty lines
                decoy_path, _, _, tm_score, _, _ = line.strip().split(maxsplit=1)[0].split(",")
                if "decoy"==decoy_path and tm_score=="tmscore":
                    continue
                else:
                    key = decoy_path.split("/")[-1].split(".")[0]
                    data[key] = float(tm_score)
    return data


def get_native_pdb_file(native_base_path: str,
                        pdb_id_list: list[str]):
    """
    Map PDB ID -> native PDB file path by taking the first file under {base}/{id}/native/.
    """
    native_id_path_mapping = {}
    for id in pdb_id_list:
        pdb_folder_path = os.path.join(native_base_path, id, "native")
        native_id_path_mapping[id] = os.path.join(pdb_folder_path, os.listdir(pdb_folder_path)[0])
    return native_id_path_mapping


def gen_xtc_except_gt(xtc_base_path: str,
                      mdthods: list[str],
                      method_decoys_path_mapping: dict[str, str],
                      xtc_folders: list[str],
                      data_native_pdb_mapping: dict[str, str],
                      pdb_id_list: list[str]
                      ):
    """
    Build XTCs for all *non-GT* methods by:
      1) Reading TM-scores,
      2) Filtering decoys by length consistency vs native sequence,
      3) Saving trajectories (all and TM>=0.5).
    """
    xtc_save_path_mapping = {method: os.path.join(xtc_base_path, xtc_folders[method_idx]) for method_idx, method in enumerate(mdthods)}

    for k, v in method_decoys_path_mapping.items():
        if not os.path.exists(v):
            os.mkdir(v)

    for k, v in method_decoys_path_mapping.items():
        xtc_save_path = xtc_save_path_mapping[k]
        print(f"=================================Method: {k}=================================")
        for pdb_id in os.listdir(v):
            if pdb_id not in pdb_id_list:
                continue
            print(f"##########################pdb_id: {pdb_id}##########################")
            decoys_folder_path = os.path.join(v, pdb_id)
            xtc_save_path_pdb = os.path.join(xtc_save_path, pdb_id)
            tm_score_path = os.path.join(v, f"{pdb_id}_decoys.csv")
            if not os.path.exists(tm_score_path):
                continue

            if not os.path.exists(xtc_save_path_pdb):
                os.mkdir(xtc_save_path_pdb)

            tm_score_mapping = parse_file_tm_scores(tm_score_path)
            sorted_pdb_files = dict(sorted(tm_score_mapping.items(), key=lambda item: item[1], reverse=True))
            sorted_keys = [k for k, v in sorted(tm_score_mapping.items(), key=lambda item: item[1], reverse=True)][:3000]
            sorted_pdb_files = {k: sorted_pdb_files[k] for k in sorted_keys}
            with open(os.path.join(xtc_save_path_pdb, f"all_sorted_keys_scores.pkl"), "wb") as f:
                pickle.dump(sorted_keys, f)
                pickle.dump(sorted_pdb_files, f)

            gt_sequences = extract_sequence_from_pdb(data_native_pdb_mapping[pdb_id])

            topology_path = os.path.join(xtc_save_path_pdb, "topology.pdb")

            position_list = []
            filterd_all_keys = []
            filtered_all_k_v_mapping = {}
            unmatching_counts = 0
            topology_flag = False
            for idx, key in tqdm(enumerate(sorted_keys)):
                pdb_file_path = os.path.join(decoys_folder_path, f"{key}.pdb")
                ca_counts = extract_coordinates_from_pdb(pdb_file_path, "CA")

                if len(ca_counts) == len(gt_sequences) and not topology_flag:
                    shutil.copy(pdb_file_path, topology_path)
                    topology_flag = True
                if len(ca_counts) != len(gt_sequences):
                    unmatching_counts += 1
                    continue
                try:
                    position = extract_coordinates_from_pdb(pdb_file_path)
                except Exception as e:
                    print(pdb_file_path)
                    print(e)
                    continue
                filterd_all_keys.append(key)
                filtered_all_k_v_mapping[key] = sorted_pdb_files[key]
                position_list.append(position)

            with open(os.path.join(xtc_save_path_pdb, f"all_sorted_keys_scores_filtered.pkl"), "wb") as f:
                pickle.dump(filterd_all_keys, f)
                pickle.dump(filtered_all_k_v_mapping, f)

            print(f"unmatching counts for all pdbs: {unmatching_counts}, positions list count: {len(position_list)}")
            save_xtc_file(topology_path, position_list, xtc_save_path_pdb, pdb_id,
                          model_type=k,
                          filter_samples=False)

            position_list = []
            filterd_keys = []
            filtered_k_v_mapping = {}
            unmatching_counts = 0
            for idx, key in enumerate(sorted_keys):
                pdb_file_path = os.path.join(decoys_folder_path, f"{key}.pdb")
                ca_counts = extract_coordinates_from_pdb(pdb_file_path, "CA")
                if sorted_pdb_files[key] <0.5:
                    continue
                if len(ca_counts) != len(gt_sequences):
                    unmatching_counts += 1
                    continue
                try:
                    position = extract_coordinates_from_pdb(pdb_file_path)
                except Exception as e:
                    print(pdb_file_path)
                    print(e)
                    continue
                filterd_keys.append(key)
                filtered_k_v_mapping[key] = sorted_pdb_files[key]
                position_list.append(position)

            with open(os.path.join(xtc_save_path_pdb, f"all_sorted_keys_scores_filtered_0_5.pkl"), "wb") as f:
                pickle.dump(filterd_keys, f)
                pickle.dump(filtered_k_v_mapping, f)

            print(f"unmatching counts for all pdbs: {unmatching_counts}, positions list count: {len(position_list)}")

            if len(position_list) == 0:
                print(f"maximum tm-score: {sorted_pdb_files[sorted_keys[0]]}, "
                      f"minimum tm-score: {sorted_pdb_files[sorted_keys[-1]]}")
            else:
                save_xtc_file(topology_path, position_list, xtc_save_path_pdb, pdb_id,
                              model_type=k,
                              filter_samples=False,
                              threashod=0.5)


def gen_xtc_gt(pdb_id_list_path: str,
               pdb_id_list: list[str],
               native_file_path_mapping: dict[str, str],
               tm_exec_path: str,
               save_path: str,
               filter_samples: bool=False,
               gt_shape_len_mapping: dict[str, int]=None
               ):
    """
    Build GT XTCs from ground-truth decoys (native-based MD or curated sets),
    compute TM-scores vs native, sort, and save (all and TM>=0.5).
    """

    for pdb_id in os.listdir(pdb_id_list_path):
        if pdb_id not in pdb_id_list:
            continue
        if pdb_id not in gt_shape_len_mapping:
            continue
        print(f"====================={pdb_id}========================")
        pdb_gt_path = os.path.join(pdb_id_list_path, pdb_id, "decoys_MD")
        gt_pdb_file_pathes = get_all_decoy_file_path(pdb_gt_path, pdb_id, gt_decoy_path=True)[pdb_id]
        gt_conformation_mapping = {key.split("/")[-1][:-4]: key for key in gt_pdb_file_pathes}

        native_pdb_path = native_file_path_mapping[pdb_id]

        pdb_save_path = os.path.join(save_path, pdb_id)
        if not os.path.exists(pdb_save_path):
            os.mkdir(pdb_save_path)

        if os.path.exists(os.path.join(pdb_save_path, "sorted_pdb_key_files_all.pkl")):
            with open(os.path.join(pdb_save_path, "sorted_pdb_key_files_all.pkl"), "rb") as f:
                sorted_keys = pickle.load(f)
                sorted_pdb_files = pickle.load(f)
        else:
            results = compute_zhang_tmscores_folder(tm_exec_path, native_pdb_path, gt_pdb_file_pathes)
            sorted_pdb_files = dict(sorted(results.items(), key=lambda item: item[1], reverse=True))
            sorted_keys = [k for k, v in sorted(results.items(), key=lambda item: item[1], reverse=True)]

            with open(os.path.join(pdb_save_path, "sorted_pdb_key_files_all.pkl"), "wb") as f:
                pickle.dump(sorted_keys, f)
                pickle.dump(sorted_pdb_files, f)

        topology_path = os.path.join(pdb_save_path, "topology_gt.pdb")
        gt_sequences = extract_sequence_from_pdb(native_pdb_path)
        native_position = extract_coordinates_from_pdb(native_pdb_path)

        position_list = []
        filtered_keys = []
        filtered_all_k_v_mapping = {}
        unmatching_counts = 0
        topology_flag = False
        for idx, key in enumerate(sorted_keys):
            pdb_file_path = gt_conformation_mapping[key]

            ca_counts = extract_coordinates_from_pdb(pdb_file_path, "CA")
            try:
                position = extract_coordinates_from_pdb(pdb_file_path)
            except Exception as e:
                print(pdb_file_path)
                print(e)
                continue

            if gt_shape_len_mapping[pdb_id] == position.shape[0] and not topology_flag:
                shutil.copy(pdb_file_path, topology_path)
                topology_flag = True
            if gt_shape_len_mapping[pdb_id] != position.shape[0]:
                unmatching_counts += 1
                continue
            if len(ca_counts) != len(gt_sequences):
                unmatching_counts += 1
                continue
            filtered_keys.append(key)
            filtered_all_k_v_mapping[key] = sorted_pdb_files[key]
            position_list.append(position)
        with open(os.path.join(pdb_save_path, f"all_sorted_keys_scores_filtered.pkl"), "wb") as f:
            pickle.dump(filtered_keys, f)
            pickle.dump(filtered_all_k_v_mapping, f)
        print(f"unmatching counts for all pdbs: {unmatching_counts}, positions list count: {len(position_list)}")

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
        xtc_path = os.path.join(pdb_save_path, f"gt_{pdb_id}.xtc")
        traj.save_xtc(xtc_path)

        position_list = []
        filterd_keys = []
        filtered_k_v_mapping = {}
        unmatching_counts = 0
        for idx, key in enumerate(sorted_keys):
            pdb_file_path = gt_conformation_mapping[key]
            tm_score = sorted_pdb_files[key]
            if tm_score < 0.5:
                continue
            ca_counts = extract_coordinates_from_pdb(pdb_file_path, "CA")
            try:
                position = extract_coordinates_from_pdb(pdb_file_path)
            except Exception as e:
                print(pdb_file_path)
                print(e)
                continue
            if gt_shape_len_mapping[pdb_id] != position.shape[0]:
                unmatching_counts += 1
                continue
            if len(ca_counts) != len(gt_sequences):
                unmatching_counts += 1
                continue
            filterd_keys.append(key)
            filtered_k_v_mapping[key] = sorted_pdb_files[key]
            position_list.append(position)
        with open(os.path.join(pdb_save_path, f"all_sorted_keys_scores_filtered_0_5.pkl"), "wb") as f:
            pickle.dump(filterd_keys, f)
            pickle.dump(filtered_k_v_mapping, f)
        print(f"unmatching counts for all pdbs: {unmatching_counts}, positions list count: {len(position_list)}")
        if len(position_list) == 0:
            print(f"maximum tm-score: {sorted_pdb_files[sorted_keys[0]]}, "
                  f"minimum tm-score: {sorted_pdb_files[sorted_keys[-1]]}")
        else:
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
            xtc_path = os.path.join(pdb_save_path, f"gt_{pdb_id}_0_5.xtc")
            traj.save_xtc(xtc_path)


def gen_gt_pca_components(pdb_id_list_path: str,
                          pdb_id_list: list[str],
                          save_path: str
                          ):
    """
    Compute PCA basis (Universe + PCA object) from GT trajectories and cache to disk.
    """
    for pdb_id in os.listdir(pdb_id_list_path):
        if pdb_id in pdb_id_list:
            pca_save_path = os.path.join(save_path, pdb_id)
            if not os.path.exists(pca_save_path):
                os.mkdir(pca_save_path)

            src_file_path = os.path.join(pdb_id_list_path, pdb_id)
            topology_file_path = os.path.join(src_file_path, "topology_gt.pdb")
            xtc_file_path = os.path.join(src_file_path, f"gt_{pdb_id}.xtc")
            xtc_0_5_file_path = os.path.join(src_file_path, f"gt_{pdb_id}_0_5.xtc")

            u, pc = get_pca_components(topology_file_path, xtc_file_path)
            with open(os.path.join(pca_save_path, "pca_components_gt.pkl"), "wb") as f:
                pickle.dump(u, f)
                pickle.dump(pc, f)

            if os.path.exists(xtc_0_5_file_path):
                u_0_5, pc_0_5 = get_pca_components(topology_file_path, xtc_0_5_file_path)
                with open(os.path.join(pca_save_path, "pca_components_gt_0_5.pkl"), "wb") as f:
                    pickle.dump(u_0_5, f)
                    pickle.dump(pc_0_5, f)



def gen_native_xtc_file(data_native_pdb_mapping: dict[str, str],
                        save_path: str):
    """
    Save a 1-frame XTC for each native PDB (for 'native star' projection).
    """
    os.makedirs(save_path, exist_ok=True)
    for key, v in data_native_pdb_mapping.items():
        positions = []
        native_position = extract_coordinates_from_pdb(v)

        gt_native_pdb_path = os.path.join(save_path, key)
        if not os.path.exists(gt_native_pdb_path):
            os.mkdir(gt_native_pdb_path)
        shutil.copy(v, f"{gt_native_pdb_path}/{key}.pdb")
        positions.append(native_position)

        topology = mdtraj.load_topology(f"{gt_native_pdb_path}/{key}.pdb")
        # Convert positions back to nm for saving to xtc.
        traj = mdtraj.Trajectory(xyz=np.stack(positions) * 0.1, topology=topology)

        traj.superpose(reference=traj, frame=0)
        xtc_path = os.path.join(gt_native_pdb_path, f"gt_{key}.xtc")
        traj.save_xtc(xtc_path)


def project_conformation(u, pc):
    """
    Project an MDAnalysis Universe onto the first two components of a pre-fit PCA.
    """

    backbone = u.select_atoms('name CA')
    transformed = pc.transform(backbone, n_components=2)
    df = pd.DataFrame(transformed, columns=['PC{}'.format(i + 1) for i in range(2)])
    df['Time (ps)'] = df.index * u.trajectory.dt
    return df


def main_component_mapping(pca_component_file_path: str,
                           xtc_root_path: str,
                           xtc_folders: list[str],
                           mdthods: list[str],
                           save_path: str
                           ):
    """
    Use GT PCA bases to project all method trajectories, compute FELs, and cache them.
    """

    for pdb_id in os.listdir(pca_component_file_path):
        pca_component_file_folder = os.path.join(pca_component_file_path, pdb_id)
        pca_component_gt_path = os.path.join(pca_component_file_folder, "pca_components_gt.pkl")
        with open(pca_component_gt_path, "rb") as f:
            u = pickle.load(f)
            pc = pickle.load(f)

        ref = u.select_atoms('name CA')

        for idx, folder in enumerate(xtc_folders):
            method_file_path = os.path.join(xtc_root_path, folder)

            if not os.path.exists(f"{method_file_path}/{pdb_id}"):
                continue
            xtc_file_path = os.path.join(f"{method_file_path}/{pdb_id}/{mdthods[idx]}_{pdb_id}.xtc")
            topology_file_path = os.path.join(f"{method_file_path}/{pdb_id}/topology.pdb")
            u_tmp = mda.Universe(topology_file_path, xtc_file_path)
            align.AlignTraj(u_tmp, ref, select='name CA', in_memory=True).run()
            pd_tmp = project_conformation(u_tmp, pc)
            pd_tmp_np = pd_tmp.to_numpy()[:, :-1]
            x, y, energy, min_val = compute_fel(pd_tmp_np)

            save_folder_path = os.path.join(save_path, mdthods[idx])
            if not os.path.exists(save_folder_path):
                os.mkdir(save_folder_path)

            save_folder_path = os.path.join(save_path, mdthods[idx], pdb_id)
            if not os.path.exists(save_folder_path):
                os.mkdir(save_folder_path)

            save_path_method = os.path.join(save_path, mdthods[idx], pdb_id, "energy.pkl")
            with open(save_path_method, "wb") as f:
                pickle.dump(pd_tmp, f)
                pickle.dump(pd_tmp_np, f)
                pickle.dump((x, y, energy, min_val), f)

        pca_component_gt_0_5_path = os.path.join(pca_component_file_folder, "pca_components_gt_0_5.pkl")
        if os.path.exists(pca_component_gt_0_5_path):
            with open(pca_component_gt_0_5_path, "rb") as f:
                u_0_5 = pickle.load(f)
                pc_0_5 = pickle.load(f)
            ref = u_0_5.select_atoms('name CA')

            for idx, folder in enumerate(xtc_folders):
                method_file_path = os.path.join(xtc_root_path, folder)
                xtc_file_path = os.path.join(f"{method_file_path}/{pdb_id}/{mdthods[idx]}_{pdb_id}_0_5.xtc")
                if not os.path.exists(xtc_file_path):
                    continue

                topology_file_path = os.path.join(f"{method_file_path}/{pdb_id}/topology.pdb")
                u_tmp = mda.Universe(topology_file_path, xtc_file_path)
                align.AlignTraj(u_tmp, ref, select='name CA', in_memory=True).run()
                pd_tmp = project_conformation(u_tmp, pc_0_5)
                pd_tmp_np = pd_tmp.to_numpy()[:, :-1]
                x, y, energy, min_val = compute_fel(pd_tmp_np)

                save_path_method = os.path.join(save_path, mdthods[idx], pdb_id, "energy_0_5.pkl")
                with open(save_path_method, "wb") as f:
                    pickle.dump(pd_tmp, f)
                    pickle.dump(pd_tmp_np, f)
                    pickle.dump((x, y, energy, min_val), f)



def main_component_gt_mapping(pca_component_file_path: str,
                              gt_xtc_file_path: str,
                              save_path: str
                              ):
    """
    Project the 1-frame native XTCs onto GT PCA to get native star coordinates (& FEL).
    """
    for pdb_id in os.listdir(pca_component_file_path):
        pca_component_file_folder = os.path.join(pca_component_file_path, pdb_id)
        pca_component_gt_path = os.path.join(pca_component_file_folder, "pca_components_gt.pkl")
        with open(pca_component_gt_path, "rb") as f:
            u = pickle.load(f)
            pc = pickle.load(f)

        ref = u.select_atoms('name CA')

        native_xtc_file_path = os.path.join(gt_xtc_file_path, pdb_id, f"gt_{pdb_id}.xtc")
        topology_file_path = os.path.join(gt_xtc_file_path, pdb_id, f"{pdb_id}.pdb")
        u_tmp = mda.Universe(topology_file_path, native_xtc_file_path)
        align.AlignTraj(u_tmp, ref, select='name CA', in_memory=True).run()
        pd_tmp = project_conformation(u_tmp, pc)
        pd_tmp_np = pd_tmp.to_numpy()[:, :-1]
        x, y, energy, min_val = compute_fel(pd_tmp_np)

        save_folder_path = os.path.join(save_path, pdb_id)
        if not os.path.exists(save_folder_path):
            os.mkdir(save_folder_path)

        save_path_method = os.path.join(save_folder_path, "energy.pkl")
        with open(save_path_method, "wb") as f:
            pickle.dump(pd_tmp, f)
            pickle.dump(pd_tmp_np, f)
            pickle.dump((x, y, energy, min_val), f)

        pca_component_gt_0_5_path = os.path.join(pca_component_file_folder, "pca_components_gt_0_5.pkl")
        if os.path.exists(pca_component_gt_0_5_path):
            with open(pca_component_gt_0_5_path, "rb") as f:
                u_0_5 = pickle.load(f)
                pc_0_5 = pickle.load(f)
            ref = u_0_5.select_atoms('name CA')
            native_xtc_file_path = os.path.join(gt_xtc_file_path, pdb_id, f"gt_{pdb_id}.xtc")
            topology_file_path = os.path.join(gt_xtc_file_path, pdb_id, f"{pdb_id}.pdb")
            u_tmp = mda.Universe(topology_file_path, native_xtc_file_path)
            align.AlignTraj(u_tmp, ref, select='name CA', in_memory=True).run()
            pd_tmp = project_conformation(u_tmp, pc_0_5)
            pd_tmp_np = pd_tmp.to_numpy()[:, :-1]
            x, y, energy, min_val = compute_fel(pd_tmp_np)

            save_path_method = os.path.join(save_folder_path, "energy_0_5.pkl")
            with open(save_path_method, "wb") as f:
                pickle.dump(pd_tmp, f)
                pickle.dump(pd_tmp_np, f)
                pickle.dump((x, y, energy, min_val), f)


def draw_image(
        pca_component_file_path: str,
        pdb_id_list: list[str],
        methods_file_path: list[str],
        benchmark_pca_component_file_path: str,
        native_energy_file_path: str,
):
    """
    Load cached FELs for each method & target, normalize min-density across methods,
    and produce comparative 2D plots (and 'near-native' plots when available).
    """
    for pdb_id in pdb_id_list:
        method_energy_landscape_mapping = {}
        method_energy_0_5_landscape_mapping = {}
        pd_np_mapping = {}
        pd_0_5_np_mapping = {}
        minval_list = []
        minval_0_5_list = []
        pca_component_file_folder = os.path.join(pca_component_file_path, pdb_id, "pca_components_gt.pkl")
        pca_component_0_5_file_folder = os.path.join(pca_component_file_path, pdb_id, "pca_components_gt_0_5.pkl")
        with open(pca_component_file_folder, "rb") as f:
            u = pickle.load(f)
            pc = pickle.load(f)
        ref = u.select_atoms('name CA')
        align.AlignTraj(u, ref, select='name CA', in_memory=True).run()
        pd = project_conformation(u, pc)
        pd_np = pd.to_numpy()[:, :-1]
        pd_np_mapping["ProteinConformers"] = pd_np
        _, _, _, min_val_gt = compute_fel(pd_np)
        minval_list.append(min_val_gt)

        if os.path.exists(pca_component_0_5_file_folder):
            with open(pca_component_0_5_file_folder, "rb") as f:
                u_0_5 = pickle.load(f)
                pc_0_5 = pickle.load(f)
            ref = u_0_5.select_atoms('name CA')
            align.AlignTraj(u_0_5, ref, select='name CA', in_memory=True).run()
            pd = project_conformation(u_0_5, pc_0_5)
            pd_0_5_np = pd.to_numpy()[:, :-1]
            _, _, _, min_val_gt_0_5 = compute_fel(pd_np)
            pd_0_5_np_mapping["ProteinConformers"] = pd_0_5_np
            minval_0_5_list.append(min_val_gt_0_5)

        for method in methods_file_path:
            energy_file_path = os.path.join(benchmark_pca_component_file_path, method, pdb_id, "energy.pkl")
            if not os.path.exists(energy_file_path):
                print(f"The file in path {energy_file_path} does not exist.")
                continue
            with open(energy_file_path, "rb") as f:
                _ = pickle.load(f)
                pd_method = pickle.load(f)
                _, _, _, min_val = pickle.load(f)
                minval_list.append(min_val)
            pd_np_mapping[method] = pd_method

            energy_0_5_file_path = os.path.join(benchmark_pca_component_file_path, method, pdb_id, "energy_0_5.pkl")
            if os.path.exists(energy_0_5_file_path):
                with open(energy_0_5_file_path, "rb") as f:
                    _ = pickle.load(f)
                    pd_0_5_method = pickle.load(f)
                    _, _, _, min_val_0_5 = pickle.load(f)
                minval_0_5_list.append(min_val_0_5)
                pd_0_5_np_mapping[method] = pd_0_5_method
        native_energy_pdb_file_path = os.path.join(native_energy_file_path, pdb_id, "energy.pkl")
        with open(native_energy_pdb_file_path, "rb") as f:
            _ = pickle.load(f)
            pd_native = pickle.load(f)
            _, _, _, min_val_native = pickle.load(f)
            pd_np_mapping["native"] = pd_native

        native_0_5_energy_file_path = os.path.join(native_energy_file_path, pdb_id, "energy_0_5.pkl")
        if os.path.exists(native_0_5_energy_file_path):
            with open(native_0_5_energy_file_path, "rb") as f:
                _ = pickle.load(f)
                pd_native_0_5 = pickle.load(f)
                pd_0_5_np_mapping["native"] = pd_native_0_5
                _, _, _, min_val_native_0_5 = pickle.load(f)

        min_val = min(minval_list)
        for k, v in pd_np_mapping.items():
            if k != "native":
                x, y, energy, _ = compute_fel(v, min_val=min_val)
                method_energy_landscape_mapping[k] = (x, y, energy)
            else:
                method_energy_landscape_mapping[k] = v
        visualize_energy_landscapes(method_energy_landscape_mapping, pdb_id)

        if len(minval_0_5_list) > 0 and len(pd_0_5_np_mapping) > 0:
            min_0_5_val = min(minval_0_5_list)
            for k, v in pd_0_5_np_mapping.items():
                if k != "native":
                    x, y, energy, _ = compute_fel(v, min_val=min_0_5_val)
                    method_energy_0_5_landscape_mapping[k] = (x, y, energy)
                else:
                    method_energy_0_5_landscape_mapping[k] = v
            visualize_energy_landscapes(method_energy_0_5_landscape_mapping, f"near_native_{pdb_id}")


def extend_and_pad(x, y, energy, global_x_min, global_x_max, global_y_min, global_y_max):
    """
    Interpolate each FEL onto a common (x,y) grid and pad missing values
    with the dataset's max energy (keeps color scale consistent).
    """
    x_new = np.linspace(global_x_min, global_x_max, 500)
    y_new = np.linspace(global_y_min, global_y_max, 500)
    X_new, Y_new = np.meshgrid(x_new, y_new)

    min_energy = np.nanmax(energy)

    # Interpolate energy onto new grid
    points = np.column_stack((x.ravel(), y.ravel()))
    energy_interp = griddata(points, energy.ravel(), (X_new, Y_new), method='linear')

    # Fill NaNs (extended regions) with minimum energy
    energy_interp[np.isnan(energy_interp)] = min_energy

    return X_new, Y_new, energy_interp


def visualize_energy_landscapes(method_energy_landscape_mapping, title_prefix):
    """
    Draw 2D FELs for multiple methods side-by-side, marking the native coordinate (*)
    from the first native frame projected into GT PCA space.
    """
    if not method_energy_landscape_mapping:
        print(f"No data to visualize for {title_prefix}")
        return

    # Compute global x/y limits
    all_x = []
    all_y = []
    all_energy = []

    native_x, native_y = method_energy_landscape_mapping.pop("native")[0]

    for key, (x, y, energy) in method_energy_landscape_mapping.items():
        all_x.extend(x.ravel())
        all_y.extend(y.ravel())
        all_energy.append(energy)

    global_x_min = np.min(all_x)
    global_x_max = np.max(all_x)
    global_y_min = np.min(all_y)
    global_y_max = np.max(all_y)
    global_vmin = np.min(all_energy)
    global_vmax = np.max(all_energy)

    # Extend and pad each dataset
    extended_data = {}
    for method, (x, y, energy) in method_energy_landscape_mapping.items():
        X_ext, Y_ext, energy_ext = extend_and_pad(
            x, y, energy,
            global_x_min, global_x_max,
            global_y_min, global_y_max
        )
        extended_data[method] = (X_ext, Y_ext, energy_ext)

    # Plotting
    num_plots = len(extended_data)
    cols = min(3, num_plots)
    rows = (num_plots + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows),
                             sharex=True, sharey=True)

    for ax, (method, (X, Y, energy)) in zip(axes.flat, extended_data.items()):
        contour = ax.contourf(X, Y, energy, levels=50, cmap='viridis',
                              vmin=global_vmin, vmax=global_vmax)
        # Add red star at native state coordinates
        ax.scatter(
            native_x, native_y,
            marker='*', s=200, color='red', edgecolor='black',
            linewidth=0.5, label='Native State', zorder=10
        )
        ax.set_title(f"{method} - {title_prefix}")
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        plt.colorbar(contour, ax=ax, label='Free Energy (kJ/mol)')

    # Hide unused subplots
    for ax in axes.flat[num_plots:]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def visualize_3d_energy_landscapes(method_energy_landscape_mapping, title_prefix):
    """
    Draw 3D surface plots of FELs with native star.
    """

    if not method_energy_landscape_mapping:
        print(f"No data to visualize for {title_prefix}")
        return

    # Extract native coordinates and remove from mapping
    native_x, native_y = method_energy_landscape_mapping.pop("native")[0]

    # Compute global x/y limits and energy range
    all_x, all_y, all_energy = [], [], []
    for key, (x, y, energy) in method_energy_landscape_mapping.items():
        all_x.extend(x.ravel())
        all_y.extend(y.ravel())
        all_energy.append(energy)

    global_x_min, global_x_max = np.min(all_x), np.max(all_x)
    global_y_min, global_y_max = np.min(all_y), np.max(all_y)
    global_vmin, global_vmax = np.min(all_energy), np.max(all_energy)

    # Extend and pad each dataset
    extended_data = {}
    for method, (x, y, energy) in method_energy_landscape_mapping.items():
        X_ext, Y_ext, energy_ext = extend_and_pad(
            x, y, energy,
            global_x_min, global_x_max,
            global_y_min, global_y_max
        )
        extended_data[method] = (X_ext, Y_ext, energy_ext)

    # Create 3D subplots
    num_plots = len(extended_data)
    cols = min(3, num_plots)
    rows = (num_plots + cols - 1) // cols

    fig = plt.figure(figsize=(8 * cols, 7 * rows))

    for idx, (method, (X, Y, energy)) in enumerate(extended_data.items()):
        ax = fig.add_subplot(rows, cols, idx + 1, projection='3d')

        # Create surface plot
        surf = ax.plot_surface(
            X, Y, energy,
            cmap='viridis',
            vmin=global_vmin,
            vmax=global_vmax,
            linewidth=0,
            antialiased=True,
            shade=True
        )

        # Find energy value at native coordinates for this method
        z_native = griddata(
            np.column_stack((X.ravel(), Y.ravel())),
            energy.ravel(),
            (native_x, native_y),
            method='linear'
        )

        # Add 3D star marker at native state
        ax.scatter(
            native_x, native_y, z_native,
            marker='*', s=300, color='red', edgecolor='black',
            linewidth=0.5, label='Native State', zorder=10
        )

        # Formatting
        ax.set_title(f"{method} - {title_prefix}")
        ax.set_xlabel('\nPrincipal Component 1', rotation=45)
        ax.set_ylabel('\nPrincipal Component 2', rotation=45)
        ax.set_zlabel('Free Energy (kJ/mol)')
        ax.view_init(elev=25, azim=-60)  # Set viewing angle

        # Add colorbar
        cbar = fig.colorbar(surf, shrink=0.5, aspect=10)
        cbar.set_label('Free Energy (kJ/mol)')

    plt.tight_layout()
    plt.show()
    

def draw_figures_with_generated_data(data_save_path: str,
                                     figure_type: str):
    """
    Utility: load pre-packed pickles {method: (X,Y,E), 'native': native_pc} and draw.
    """
    for pdb_id in os.listdir(data_save_path):
        pdb_path = os.path.join(data_save_path, pdb_id)
        real_pdb_id = pdb_id.split(".")[0]
        with open(pdb_path, 'rb') as f:
            method_energy_landscape_mapping = pickle.load(f)
            if figure_type == "3d":
                visualize_3d_energy_landscapes(method_energy_landscape_mapping, real_pdb_id)
            else:
                visualize_energy_landscapes(method_energy_landscape_mapping, real_pdb_id)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="CASP dynamics FEL pipeline")
    # Targets & methods
    p.add_argument(
        "--pdb-ids",
        nargs="+",
        default=["T1027", "T1029", "T1030", "T1031", "T1033", "T1034", "T1035", "T1037",
                 "T1039", "T1040", "T1041", "T1043", "T1052", "T1062", "T1065s2",
                 "T1087", "T1092", "T1104"],
        help="CASP target IDs to process.",
    )
    p.add_argument(
        "--methods",
        nargs="+",
        default=["alphaflow_md", "alphaflow_pdb", "bioemu", "esmflow_md", "esmflow_pdb", "esmdiff", "alphafold3"],
        help="Method names (must align with xtc-folder order).",
    )
    p.add_argument(
        "--xtc-folders",
        nargs="+",
        default=["alphaflow_md_xtcs", "alphaflow_pdb_xtcs", "bioemu_xtcs",
                 "esmflow_md_xtcs", "esmflow_pdb_xtcs", "esmdiff_xtcs", "alphafold3_xtcs"],
        help="Per-method XTC output folders, same order as --methods.",
    )

    # Decoy sources (per method)
    p.add_argument("--bioemu-decoys", help="The folder of decoys generated by bioemu", default="/mnt/dna01/library2/caspdynamics/bioemu_decoys/")
    p.add_argument("--alphaflow-md-decoys",  help="The folder of decoys generated by alphaflow md", default="/mnt/dna01/library2/caspdynamics/esmflow_decoys/casp1415_decoys_alphaflow_md_distilled/")
    p.add_argument("--alphaflow-pdb-decoys",  help="The folder of decoys generated by alphaflow pdb", default="/mnt/dna01/library2/caspdynamics/esmflow_decoys/casp1415_decoys_alphaflow_pdb_distilled/")
    p.add_argument("--esmflow-md-decoys", help="The folder of decoys generated by esmflow md", default="/mnt/dna01/library2/caspdynamics/esmflow_decoys/casp1415_decoys_esmflow_md_distilled")
    p.add_argument("--esmflow-pdb-decoys", help="The folder of decoys generated by esmflow pdb", default="/mnt/dna01/library2/caspdynamics/esmflow_decoys/casp1415_decoys_alphaflow_pdb_distilled")
    p.add_argument("--esmdiff-decoys", help="The folder of decoys generated by esmdiff", default="/mnt/dna01/library2/caspdynamics/esmdiff_decoys")
    p.add_argument("--alphafold3-decoys", help="The folder of decoys generated by af3", default="/mnt/dna01/library2/caspdynamics/esmdiff_decoys")

    # Paths for native/outputs
    p.add_argument("--native-file-base", default="/mnt/rna01/zyh/data/selected_database2", help="Native protein structure folder path")
    p.add_argument("--xtc-base", default="/mnt/dna01/library2/caspdynamics/xtcs_new", help="XTCs save path.")
    p.add_argument("--xtc-gt-save", default="/mnt/dna01/library2/caspdynamics/xtcs/gt", help="XTCs native structure save path.")
    p.add_argument("--pca-component-dir", default="/mnt/dna01/library2/caspdynamics/xtcs/gt_pca_folder", help="PCA components save path.")
    p.add_argument("--xtc-gt-native-save", default="/mnt/dna01/library2/caspdynamics/xtcs/gt_native_xtc", help="The native structure's pca components save path.")
    p.add_argument("--energy-dir", default="/mnt/dna01/library2/caspdynamics/xtcs/energys_new", help="Energy save path.")
    p.add_argument("--energy-native-mapping-dir", default="/mnt/dna01/library2/caspdynamics/xtcs/energys/native_mapping", help="Energy of the native structure.")

    # Executables / misc
    p.add_argument("--tm-score-exe", default="/mnt/rna01/chenw/Software/TMscore", help="Path to TMscore binary.")
    p.add_argument("--filter-samples", action="store_true", help="Filter unphysical frames when saving XTCs.")
    return p


def main():
    args = build_argparser().parse_args()

    # Basic sanity checks
    if len(args.methods) != len(args.xtc_folders):
        raise ValueError("--methods and --xtc-folders must have the same length")

    pdb_id_list = list(args.pdb_ids)

    # Per-method decoy roots
    method_decoys_path_mapping = {
        "alphaflow_md": args.alphaflow_md_decoys,
        "alphaflow_pdb": args.alphaflow_pdb_decoys,
        "bioemu": args.bioemu_decoys,
        "esmflow_md": args.esmflow_md_decoys,
        "esmflow_pdb": args.esmflow_pdb_decoys,
        "esmdiff": args.esmdiff_decoys,
        "alphafold3": args.alphafold3_decoys,
    }

    # Native PDB mapping (id -> file)
    data_native_pdb_mapping = get_native_pdb_file(args.native_file_base, pdb_id_list)

    # Step 2. generate trajectories of benchmark methods (non-GT)
    gen_xtc_except_gt(
        xtc_base_path=args.xtc_base,
        mdthods=args.methods,
        method_decoys_path_mapping=method_decoys_path_mapping,
        xtc_folders=args.xtc_folders,
        data_native_pdb_mapping=data_native_pdb_mapping,
        pdb_id_list=pdb_id_list,
    )

    # Step 3. generate ground truth benchmark trajectories
    gt_shape_len_mapping = {
        "T1027": 2565, "T1029": 2001, "T1039": 2651, "T1037": 6350, "T1104": 1805,
        "T1041": 3828, "T1092": 6973, "T1030": 4390, "T1031": 1558, "T1087": 1393,
        "T1035": 1666, "T1062": 510,  "T1033": 1618, "T1034": 2500, "T1040": 2092,
        "T1065s2": 1542, "T1043": 2392, "T1052": 12328
    }

    gen_xtc_gt(
        pdb_id_list_path=args.native_file_base,   # source root with per-ID folders
        pdb_id_list=pdb_id_list,
        native_file_path_mapping=data_native_pdb_mapping,
        tm_exec_path=args.tm_score_exe,
        save_path=args.xtc_gt_save,               # where GT xtcs are written
        filter_samples=args.filter_samples,
        gt_shape_len_mapping=gt_shape_len_mapping,
    )

    # Step 4. generate PCA components from GT
    gen_gt_pca_components(
        pdb_id_list_path=args.xtc_gt_save,
        pdb_id_list=pdb_id_list,
        save_path=args.pca_component_dir,
    )

    # Step 5. generate native (1-frame) XTCs
    gen_native_xtc_file(
        data_native_pdb_mapping,
        save_path=args.xtc_gt_native_save,
    )

    # Step 6. method PCA projections + FELs
    main_component_mapping(
        pca_component_file_path=args.pca_component_dir,
        xtc_root_path=args.xtc_base,
        xtc_folders=args.xtc_folders,
        mdthods=args.methods,
        save_path=args.energy_dir,
    )

    # Step 7. native PCA projections (for native star)
    main_component_gt_mapping(
        pca_component_file_path=args.pca_component_dir,
        gt_xtc_file_path=args.xtc_gt_native_save,
        save_path=args.energy_native_mapping_dir,
    )

    # Step 8. comparative plots (2D FELs; near-native if available)
    draw_image(
        pca_component_file_path=args.pca_component_dir,
        pdb_id_list=pdb_id_list,
        methods_file_path=args.methods,
        benchmark_pca_component_file_path=args.energy_dir,
        native_energy_file_path=args.energy_native_mapping_dir,
    )


if __name__ == "__main__":
    main()