import os
import shutil
import mdtraj
import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis import rms, align, pca
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append("/home/bingwu/WorkSpaces/ProteinDynamicBenchmark")

from pdbench.utils.data_utils import get_all_decoy_file_path
from bioemu.convert_chemgraph import filter_unphysical_traj
from pdbench.utils.data_utils import (extract_coordinates_from_pdb, save_xtc_file,
                                      generate_xtc, extract_sequence_from_pdb)


def generate_esmdiff_xtc(
        native_file_path: str,
        esmdiff_pdb_file_path: str,
        tm_exec_path: str,
        esmdiff_save_path: str,
        pdb_id: str,
        filter_samples: bool = True
):
    sorted_pdb_files, sorted_keys = generate_xtc(native_file_path,
                                                 esmdiff_pdb_file_path,
                                                 tm_exec_path,
                                                 esmdiff_save_path,
                                                 "esmdiff")

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

    save_xtc_file(topology_path, position_list, esmdiff_save_path, pdb_id,
                  model_type="esmdiff",
                  filter_samples=filter_samples)


def generate_bioemu_xtc(native_file_path,
                        bioemu_pdbs: str,
                        tm_exec_path: str,
                        bioemu_save_path: str,
                        pdb_id: str,
                        filter_samples: bool = True
                        ):
    sorted_pdb_files, sorted_keys = generate_xtc(native_file_path,
                                                 bioemu_pdbs,
                                                 tm_exec_path,
                                                 bioemu_save_path,
                                                 "bioemu")
    topology_path = os.path.join(bioemu_save_path, "topology_bioemu.pdb")

    position_list = []
    for idx, key in enumerate(sorted_keys):
        pdb_file_path = os.path.join(bioemu_pdbs, f"{key}")
        if idx == 0:
            shutil.copy(pdb_file_path, topology_path)
            shutil.copy(pdb_file_path, topology_path)
        try:
            position = extract_coordinates_from_pdb(pdb_file_path)
        except Exception as e:
            print(pdb_file_path)
            print(e)
            continue
        position_list.append(position)

    save_xtc_file(topology_path, position_list, bioemu_save_path, pdb_id,
                  model_type="bioemu",
                  filter_samples=filter_samples)


def generate_gt_xtc(
        native_file_path: str,
        gt_file_path: str,
        tm_exec_path: str,
        gt_save_path="/home/bingwu/Downloads/Dynamics/T1024_xtc/gt/T1024/",
        pdb_id="T1024",
        filter_samples: bool = True
):
    gt_file_list = get_all_decoy_file_path(gt_file_path, pdb_id, gt_decoy_path=True)[pdb_id]
    gt_conformation_mapping = {key.split("/")[-1][:-4]: key for key in gt_file_list}

    # sorted_pdb_files, sorted_keys = generate_xtc(native_file_path,
    #                                              gt_file_list,
    #                                              tm_exec_path,
    #                                              gt_save_path,
    #                                              "gt")

    with open("/home/bingwu/Downloads/Conformation_Generation_Project/Dynamics/xtc_new/gt/gt/gt_tm_score.pkl", "rb") as f:
        sorted_pdb_files = pickle.load(f)
        sorted_keys = pickle.load(f)

    topology_path = os.path.join(gt_save_path, "topology_gt.pdb")

    position_list = []
    copy_flag = False
    filtered_keys = []
    for idx, key in enumerate(sorted_keys):
        pdb_file_path = gt_conformation_mapping[key]
        position = extract_coordinates_from_pdb(pdb_file_path)
        if position.shape[0] != 6490:
            continue
        if not copy_flag:
            shutil.copy(pdb_file_path, topology_path)
            copy_flag = True
        filtered_keys.append(key)
        position_list.append(position)
    # position_list = [position for position in position_list if position.shape[0] == 6490]

    with open(f"{gt_save_path}/gt_filtered_keys_{pdb_id}.pkl", "wb") as f:
        pickle.dump(filtered_keys, f)

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


def generate_gt_xtc_filter(
        native_file_path: str,
        gt_file_path: str,
        tm_exec_path: str,
        gt_save_path="/home/bingwu/Downloads/Dynamics/T1024_xtc/gt/T1024/",
        pdb_id="T1024",
        thr: float = 0.5,
        filter_samples: bool = True
):
    gt_file_list = get_all_decoy_file_path(gt_file_path, pdb_id, gt_decoy_path=True)[pdb_id]
    gt_conformation_mapping = {key.split("/")[-1][:-4]: key for key in gt_file_list}

    # sorted_pdb_files, sorted_keys = generate_xtc(native_file_path,
    #                                              gt_file_list,
    #                                              tm_exec_path,
    #                                              gt_save_path,
    #                                              "gt")

    with open("/home/bingwu/Downloads/Conformation_Generation_Project/Dynamics/xtc_new/gt/gt/gt_tm_score.pkl", "rb") as f:
        sorted_pdb_files = pickle.load(f)
        sorted_keys = pickle.load(f)

    topology_path = os.path.join(gt_save_path, "topology_gt.pdb")

    position_list = []
    copy_flag = False
    filtered_keys = []
    filtered_pdb_mapping = {}
    for idx, key in enumerate(sorted_keys):
        tm_score = sorted_pdb_files[key]
        if tm_score < thr:
            continue
        pdb_file_path = gt_conformation_mapping[key]
        position = extract_coordinates_from_pdb(pdb_file_path)
        if position.shape[0] != 6490:
            continue
        if not copy_flag:
            shutil.copy(pdb_file_path, topology_path)
            copy_flag = True
        filtered_keys.append(key)
        filtered_pdb_mapping[key] = sorted_pdb_files[key]
        position_list.append(position)
    # position_list = [position for position in position_list if position.shape[0] == 6490]

    with open(f"{gt_save_path}/gt_filtered_keys_{pdb_id}.pkl", "wb") as f:
        pickle.dump(filtered_keys, f)
        pickle.dump(filtered_pdb_mapping, f)

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



def get_pca_components(topology_pdb_path: str,
                       trajectory_xtc_path: str,
                       ref=None):
    u = mda.Universe(topology_pdb_path, trajectory_xtc_path)
    if ref is None:
        ref = u.select_atoms('backbone')  # options: "protein", "name CA", "backbone"
    align.AlignTraj(u, ref, select='name CA', in_memory=True).run()
    pc = pca.PCA(u, select='name CA', align=True, mean=None,
                   n_components=None).run()
    return u, pc


def project_conformation(u, pc):
    backbone = u.select_atoms('name CA')
    transformed = pc.transform(backbone, n_components=2)
    df = pd.DataFrame(transformed, columns=['PC{}'.format(i + 1) for i in range(2)])
    df['Time (ps)'] = df.index * u.trajectory.dt
    return df


def write_components2file(components_obj, save_path: str):
    with open(save_path, 'w') as f:
        f.write("# graph.xvg for g_sham\n")
        for index, row in components_obj.iterrows():
            time = row['Time (ps)']
            pc1 = row['PC1']
            pc2 = row['PC2']
            f.write(f"{time:.3f}\t{pc1:.3f}\t{pc2:.3f}\n")


def compute_fel(cvs, bins=50, kT=2.494, min_val=0.0):
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
    if min_val == 0.0:
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

    return X, Y, pmf



if __name__ == "__main__":
    topology_pdb = "/home/bingwu/Downloads/Conformation Generation Project/Dynamics/T1024_xtc/gt/T1024/topology_gt.pdb"
    trajectory_xtc = "/mnt/rna01/chenw/Datasets/Protein_Dynamic/T1024_xtc/gt_new/T1024/gt_T1024.xtc"

    tm_exec_path = "/home/bingwu/Softwares/TMscore"
    native_pdb_path = "/home/bingwu/Downloads/Conformation_Generation_Project/Dynamics/T1024_GT/native/T1024.pdb"

    gt_conformations_path = "/home/bingwu/Downloads/Conformation_Generation_Project/Dynamics/T1024_GT/decoys_MD"
    esmdiff_conformations_path = "/home/bingwu/Downloads/Conformation_Generation_Project/Dynamics/T1024_esmdiff"
    bioemu_conformation_pdb_sequence_path = "/home/bingwu/Downloads/Conformation_Generation_Project/Dynamics/T1024_bioemu_pdb_sequence"
    bioemu_conformation_native_sequence_path = "/home/bingwu/Downloads/Conformation_Generation_Project/Dynamics/T1024_bioemu_native_sequence"

    gt_topology_pdb_path = "/home/bingwu/Downloads/Conformation_Generation_Project/Dynamics/xtc_new/gt/gt/topology_gt.pdb"
    gt_xtc_file_path = "/home/bingwu/Downloads/Conformation_Generation_Project/Dynamics/xtc_new/gt/gt/gt_T1024.xtc"
    u, pc = get_pca_components(gt_topology_pdb_path, gt_xtc_file_path)
    pd_gt = project_conformation(u, pc)
    ref = u.select_atoms('name CA')
    #
    topology_pdb_bioemu_native = "/home/bingwu/Downloads/Conformation_Generation_Project/Dynamics/xtc_new/bioemu_native_sequence/topology_esmdiff.pdb"
    trajectory_xtc_bioemu_native = "/home/bingwu/Downloads/Conformation_Generation_Project/Dynamics/xtc_new/bioemu_native_sequence/esmdiff_T1024.xtc"
    u_bioemu_native = mda.Universe(topology_pdb_bioemu_native, trajectory_xtc_bioemu_native)
    aligner_bioemu = align.AlignTraj(u_bioemu_native, ref, select='name CA', in_memory=True).run()
    pd_bioemu_native = project_conformation(u_bioemu_native, pc)
    # #
    topology_pdb_pdb_bioemu = "/home/bingwu/Downloads/Conformation_Generation_Project/Dynamics/xtc_new/esmdiff/topology_esmdiff.pdb"
    trajectory_xtc_bioemu_pdb = "/home/bingwu/Downloads/Conformation_Generation_Project/Dynamics/xtc_new/esmdiff/esmdiff_T1024.xtc"
    u_bioemu_pdb = mda.Universe(topology_pdb_pdb_bioemu, trajectory_xtc_bioemu_pdb)
    align.AlignTraj(u_bioemu_pdb, ref, select='name CA', in_memory=True).run()
    pd_esmdiff = project_conformation(u_bioemu_pdb, pc)
    #
    # write_components2file(pd_gt, "/home/bingwu/Downloads/Conformation_Generation_Project/Dynamics/pca_files/gt.xvg")
    # write_components2file(pd_bioemu_native, "/home/bingwu/Downloads/Conformation_Generation_Project/Dynamics/pca_files/bioemu_native.xvg")
    # write_components2file(pd_esmdiff, "/home/bingwu/Downloads/Conformation_Generation_Project/Dynamics/pca_files/esmdiff.xvg")
    # write_components2file(pd_bioemu_pdb, "/home/bingwu/Downloads/Conformation_Generation_Project/Dynamics/pca_files/bioemu_pdb.xvg")

    trajectory_file_path_gt = "/home/bingwu/Downloads/Conformation_Generation_Project/Dynamics/pca_files/gt.xvg"
    trajectory_file_path_bioemu = "/home/bingwu/Downloads/Conformation_Generation_Project/Dynamics/pca_files/bioemu_native.xvg"
    trajectory_file_path_esmdiff = "/home/bingwu/Downloads/Conformation_Generation_Project/Dynamics/pca_files/esmdiff.xvg"

    pd_gt_np = pd_gt.to_numpy()[:, :-1]
    pd_bioemu_native_np = pd_bioemu_native.to_numpy()[:, :-1]
    pd_esmdiff_np = pd_esmdiff.to_numpy()[:, :-1]


    X_gt, Y_gt, gt_energy = compute_fel(pd_gt_np, bins=64, kT=2.494)
    X_B, Y_B, bioemu_energy = compute_fel(pd_bioemu_native_np, bins=64, kT=2.494)
    X_E, Y_E, esmdiff_energy = compute_fel(pd_esmdiff_np, bins=64, kT=2.494)

    draw_type = "2d"

    if draw_type == "2d":
        # # plt.contourf(X_gt, Y_gt, gt_energy, levels=50, cmap='viridis')
        # plt.contourf(X_B, Y_B, bioemu_energy, levels=50, cmap='viridis')
        # # plt.contourf(X_E, Y_E, esmdiff_energy, levels=50, cmap='viridis')
        # plt.colorbar(label='Free Energy (kJ/mol)')
        # plt.xlabel('Component 1')
        # plt.ylabel('Component 2')
        # plt.title('Free Energy Landscape')
        # plt.show()
        # Assuming X_gt, Y_gt, gt_energy, X_B, Y_B, bioemu_energy, X_E, Y_E, esmdiff_energy are already defined

        # Combine all energy values to determine consistent color scale
        all_energy = np.concatenate([gt_energy.ravel(), bioemu_energy.ravel(), esmdiff_energy.ravel()])
        vmin, vmax = np.min(all_energy), np.max(all_energy)

        # Create a figure with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)

        # Coordinates for the star
        x_star = 17.169
        y_star = 10.58

        # Plot Ground Truth (GT)
        contour_gt = axes[0].contourf(X_gt, Y_gt, gt_energy, levels=50, cmap='viridis', vmin=vmin, vmax=vmax)
        axes[0].set_title('Ground Truth (GT)')
        axes[0].set_xlabel('Component 1')
        axes[0].set_ylabel('Component 2')
        # Add star
        axes[0].plot(x_star, y_star, '*', color='red', markersize=10)

        # Plot BioEmu
        contour_B = axes[1].contourf(X_B, Y_B, bioemu_energy, levels=50, cmap='viridis', vmin=vmin, vmax=vmax)
        axes[1].set_title('BioEmu')
        axes[1].set_xlabel('Component 1')
        # Add star
        axes[1].plot(x_star, y_star, '*', color='red', markersize=10)

        # Plot ESMdiff
        contour_E = axes[2].contourf(X_E, Y_E, esmdiff_energy, levels=50, cmap='viridis', vmin=vmin, vmax=vmax)
        axes[2].set_title('ESMdiff')
        axes[2].set_xlabel('Component 1')
        # Add star
        axes[2].plot(x_star, y_star, '*', color='red', markersize=10)

        # Add a single colorbar for all subplots
        cbar = fig.colorbar(contour_E, ax=axes, shrink=0.6)
        cbar.set_label('Free Energy (kJ/mol)')

        plt.tight_layout()
        plt.show()
    elif draw_type == "3d":
        fig, axes = plt.subplots(1, 3, figsize=(20, 7), subplot_kw={'projection': '3d'})

        # Plot GT
        axes[0].plot_surface(X_gt, Y_gt, gt_energy, cmap='viridis')
        axes[0].set_title('Ground Truth')

        # Plot BioEmu
        axes[1].plot_surface(X_B, Y_B, bioemu_energy, cmap='viridis')
        axes[1].set_title('BioEmu')

        # Plot ESMdiff
        axes[2].plot_surface(X_E, Y_E, esmdiff_energy, cmap='viridis')
        axes[2].set_title('ESMdiff')

        plt.tight_layout()
        plt.show()