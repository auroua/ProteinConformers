# eval_utils.py
import os
import glob
import subprocess
import tempfile
import time
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from collections import Counter # Needed for finding the mode length

import numpy as np
import pandas as pd
from scipy.spatial import distance
from scipy.stats import entropy # For KL divergence within JS
# from scipy.stats import spearmanr, pearsonr # Keep for potential future use, not used currently
from tqdm import tqdm

# --- Dependencies ---
try:
    from Bio.PDB import PDBParser, CaPPBuilder
    from Bio.SVDSuperimposer import SVDSuperimposer
except ImportError:
    print("Error: BioPython not found. Please install it: pip install biopython")
    exit()

try:
    from deeptime.decomposition import TICA
except ImportError:
    print("Warning: deeptime not found. JS-TIC metric will not be available. "
          "Install it if needed: pip install deeptime")
    TICA = None

# --- Configuration ---
_BASE_DIRS = {}

def set_base_dirs(model_decoys_root_dir: str, native_pdb_root_dir: str, gt_decoys_root_dir: str, tmalign_exec_path: str):
    """Sets up the base directories required by the evaluation utilities."""
    global _BASE_DIRS
    _BASE_DIRS["model_decoys_root"] = Path(model_decoys_root_dir)
    _BASE_DIRS["native_pdbs_root"] = Path(native_pdb_root_dir)
    _BASE_DIRS["gt_decoys_root"] = Path(gt_decoys_root_dir)
    _BASE_DIRS["tmalign_executable"] = Path(tmalign_exec_path)

def get_base_dir(key: str) -> Path:
    """Retrieves a configured base directory path."""
    if not _BASE_DIRS:
        raise ValueError("Base directories not set. Call set_base_dirs() first.")
    if key not in _BASE_DIRS:
        raise KeyError(f"Base directory key '{key}' not found in _BASE_DIRS. Available keys: {list(_BASE_DIRS.keys())}")
    return _BASE_DIRS[key]

# --- Constants ---
PSEUDO_C = 1e-6
CLASH_THRESHOLD_CA = 3.0
PW_DIST_SKIP = 3
HIST_N_BINS = 50
TICA_LAGTIME = 20

# --- Timer Decorator ---
def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Elapsed time ({func.__name__}): {elapsed_time:.2f} sec")
        # If the function itself returns a tuple (value, time), just return it
        # Otherwise, create a new tuple (value, time)
        if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], float) and result[1] < 100000: # Heuristic: time already appended
             return result
        else: # Function returned a single value or a tuple not matching (value, time)
            return (result, float(f'{elapsed_time:.2f}'))
    return wrapper


# --- Coordinate Loading (UPDATED with 2-pass strategy) ---
def load_coords_from_dir(pdb_dir: Path, recursive: bool = False) -> Tuple[List[np.ndarray], List[str]]:
    """
    Loads PDBs, extracts C-alpha coordinates, determines the consensus length,
    and returns coordinates and filenames matching that length. Handles 0-based
    vs 1-based indexing differences by focusing on the count of C-alpha atoms.
    """
    if recursive:
        pdb_files = sorted(list(pdb_dir.rglob("*.pdb")))
    else:
        pdb_files = sorted(list(pdb_dir.glob("*.pdb")))

    if not pdb_files:
        print(f"Warning: No PDB files found in {pdb_dir} {'(recursively)' if recursive else ''}")
        return [], []

    parser = PDBParser(QUIET=True)
    # ppb = CaPPBuilder() # Using direct atom iteration for CA count robustness
    temp_results = [] # Store tuples of (coords, filename, length)
    ca_lengths = []

    print(f"Pass 1: Reading {len(pdb_files)} PDB files to determine C-alpha counts in {pdb_dir} {'(recursively)' if recursive else ''}...")
    for pdb_file in tqdm(pdb_files, desc=f"Pass 1 Reading {pdb_dir.name}"):
        try:
            structure = parser.get_structure(pdb_file.stem, pdb_file)
            model = structure[0]
            chain_coords_list = []
            ca_count = 0
            for residue in model.get_residues(): # Iterate through residues
                if "CA" in residue: # Check for CA atom in residue
                    chain_coords_list.append(residue["CA"].get_coord())
                    ca_count +=1 # This count is now robust to residue numbering

            if ca_count > 0: # Equivalent to checking if chain_coords_list is not empty
                 coords_array = np.array(chain_coords_list)
                 current_length = coords_array.shape[0] # This is the actual number of CA atoms found
                 ca_lengths.append(current_length)
                 filename = str(pdb_file.relative_to(pdb_dir)) if recursive else pdb_file.name
                 temp_results.append({'coords': coords_array, 'filename': filename, 'length': current_length})

        except Exception as e:
            # print(f"Debug: Could not parse or process {pdb_file.name}: {e}. Skipping.")
            pass

    if not ca_lengths:
        print(f"Error: No valid C-alpha coordinates could be extracted from any PDB in {pdb_dir}")
        return [], []

    length_counts = Counter(ca_lengths)
    if not length_counts:
         print(f"Error: No C-alpha lengths recorded for PDBs in {pdb_dir}")
         return [], []

    target_length = length_counts.most_common(1)[0][0]
    print(f"Determined target C-alpha length for {pdb_dir.name}: {target_length} (based on {len(ca_lengths)} successfully parsed structures)")
    if len(length_counts) > 1:
        print(f"Warning: Multiple C-alpha lengths found: {dict(length_counts)}. Using the most common length ({target_length}).")

    coords_list = []
    filenames = []
    skipped_count = 0
    for result in temp_results:
        if result['length'] == target_length:
            coords_list.append(result['coords'])
            filenames.append(result['filename'])
        else:
            skipped_count += 1
    if skipped_count > 0:
         print(f"Skipped {skipped_count} structures due to C-alpha count mismatch (expected {target_length}).")
    print(f"Pass 2: Successfully loaded {len(coords_list)} structures matching target C-alpha count {target_length}.")
    if not coords_list:
        print(f"Error: Failed to load any structures matching the target C-alpha count from {pdb_dir}")
    return coords_list, filenames


# --- Feature Calculation (Unchanged) ---
def distance_matrix_ca(coords: np.ndarray) -> np.ndarray:
    assert coords.ndim == 2 and coords.shape[1] == 3, f"Expected (L, 3) coords, got {coords.shape}"
    return distance.squareform(distance.pdist(coords))

def pairwise_distance_ca(coords: np.ndarray, k_offset: int = PW_DIST_SKIP) -> np.ndarray:
    assert coords.ndim == 2 and coords.shape[1] == 3, f"Expected (L, 3) coords, got {coords.shape}"
    dist_matrix = distance_matrix_ca(coords)
    L = dist_matrix.shape[0]
    if L <= k_offset:
        return np.array([])
    indices = np.triu_indices(L, k=k_offset + 1)
    return dist_matrix[indices]

def radius_of_gyration(coords: np.ndarray) -> float:
    assert coords.ndim == 2 and coords.shape[1] == 3, f"Expected (L, 3) coords, got {coords.shape}"
    center_of_mass = np.mean(coords, axis=0)
    return np.sqrt(np.mean(np.sum((coords - center_of_mass)**2, axis=1)))

# --- Histogram and JS Divergence Helpers (Unchanged) ---
def get_histogram(data: np.ndarray, n_bins: int, data_range: Optional[Tuple[float, float]] = None) -> Tuple[np.ndarray, np.ndarray]:
    if data.size == 0:
        return np.ones(n_bins) / n_bins, np.linspace(0, 1, n_bins + 1)
    if data_range is None:
        min_val, max_val = np.min(data), np.max(data)
        data_range = (min_val, max_val)
    else:
        min_val, max_val = data_range
    if min_val == max_val:
        hist = np.zeros(n_bins)
        if n_bins > 0 : hist[n_bins // 2] = 1.0
        bin_edges = np.linspace(min_val - 0.5, max_val + 0.5, n_bins + 1)
        if n_bins == 1: bin_edges = np.array([min_val - 0.5, max_val + 0.5])
        return hist, bin_edges
    hist, bin_edges = np.histogram(data, bins=n_bins, range=data_range, density=False)
    hist = hist.astype(float) + PSEUDO_C
    hist /= np.sum(hist)
    return hist, bin_edges

def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    m = 0.5 * (p + q)
    m_safe = np.where(m == 0, 1e-10, m)
    p_safe = np.where(p == 0, 1e-10, p)
    q_safe = np.where(q == 0, 1e-10, q)
    kl_pm = entropy(p_safe, m_safe, base=2)
    kl_qm = entropy(q_safe, m_safe, base=2)
    jsd = np.clip(0.5 * (kl_pm + kl_qm), 0, None)
    return float(jsd)

# --- JS Divergence Metrics ---
@timer
def calculate_js_pwd(coords_list1: List[np.ndarray], coords_list2: List[np.ndarray],
                     n_bins: int = HIST_N_BINS, pwd_offset: int = PW_DIST_SKIP) -> float:
    if not coords_list1 or not coords_list2: return np.nan
    print("Calculating JS-PwD...")
    pwd1 = np.concatenate([pairwise_distance_ca(c, k_offset=pwd_offset) for c in coords_list1 if c.ndim==2 and c.shape[0]>pwd_offset])
    pwd2 = np.concatenate([pairwise_distance_ca(c, k_offset=pwd_offset) for c in coords_list2 if c.ndim==2 and c.shape[0]>pwd_offset])
    if pwd1.size == 0 or pwd2.size == 0: return np.nan
    combined_pwd = np.concatenate((pwd1, pwd2))
    data_range = (np.min(combined_pwd), np.max(combined_pwd))
    hist1, _ = get_histogram(pwd1, n_bins=n_bins, data_range=data_range)
    hist2, _ = get_histogram(pwd2, n_bins=n_bins, data_range=data_range)
    return js_divergence(hist1, hist2)

@timer
def calculate_js_rg(coords_list1: List[np.ndarray], coords_list2: List[np.ndarray],
                    n_bins: int = HIST_N_BINS) -> float:
    if not coords_list1 or not coords_list2: return np.nan
    print("Calculating JS-Rg...")
    rg1 = np.array([radius_of_gyration(c) for c in coords_list1 if c.ndim == 2])
    rg2 = np.array([radius_of_gyration(c) for c in coords_list2 if c.ndim == 2])
    if rg1.size == 0 or rg2.size == 0: return np.nan
    combined_rg = np.concatenate((rg1, rg2))
    data_range = (np.min(combined_rg), np.max(combined_rg))
    hist1, _ = get_histogram(rg1, n_bins=n_bins, data_range=data_range)
    hist2, _ = get_histogram(rg2, n_bins=n_bins, data_range=data_range)
    return js_divergence(hist1, hist2)

@timer # UPDATED JS-TIC
def calculate_js_tica(coords_list1: List[np.ndarray], coords_list2: List[np.ndarray],
                      n_bins: int = HIST_N_BINS, lagtime: int = TICA_LAGTIME,
                      n_components_requested: int = 2, pwd_offset_for_tica: int = 0) -> float:
    if TICA is None:
        print("Warning: deeptime not installed. Cannot calculate JS-TIC.")
        return np.nan
    if not coords_list1 or not coords_list2:
        print("Warning: Empty coordinate list provided for JS-TIC.")
        return np.nan
    
    # Ensure enough samples for the given lag time in the reference ensemble (coords_list2)
    # Number of lagged pairs is n_samples - lagtime. We need at least 1 lagged pair.
    # So, n_samples - lagtime >= 1  => n_samples > lagtime
    if len(coords_list2) <= lagtime:
        print(f"Warning: Reference ensemble size ({len(coords_list2)}) is not greater than lagtime ({lagtime}). "
              "Not enough data for TICA. Skipping JS-TIC.")
        return np.nan

    print(f"Calculating JS-TIC (fitting on ensemble 2, lag={lagtime}, pwd_offset={pwd_offset_for_tica}, requested components={n_components_requested})...")
    try:
        print("Calculating pairwise distances for TICA input...")
        pwd1_list = [pairwise_distance_ca(c, k_offset=pwd_offset_for_tica) for c in tqdm(coords_list1, desc="PwD Ens1 (TICA)") if c.ndim == 2 and c.shape[0] > pwd_offset_for_tica]
        pwd2_list = [pairwise_distance_ca(c, k_offset=pwd_offset_for_tica) for c in tqdm(coords_list2, desc="PwD Ens2 (TICA)") if c.ndim == 2 and c.shape[0] > pwd_offset_for_tica]

        if not pwd1_list or not pwd2_list:
            print("Warning: Not enough valid PwD vectors for TICA after PwD calculation. Skipping JS-TIC.")
            return np.nan

        ref_len_pwd = pwd1_list[0].shape[0] if pwd1_list else (pwd2_list[0].shape[0] if pwd2_list else 0)
        if ref_len_pwd == 0 and (len(pwd1_list) > 0 or len(pwd2_list) > 0) : # PwD resulted in empty arrays for all structures
             first_coords_len = coords_list1[0].shape[0] if coords_list1 else (coords_list2[0].shape[0] if coords_list2 else 0)
             print(f"Warning: Pairwise distance vectors are empty (e.g. structure length {first_coords_len} <= pwd_offset {pwd_offset_for_tica}). Skipping JS-TIC.")
             return np.nan


        pwd1_filtered = [p for p in pwd1_list if p.shape[0] == ref_len_pwd]
        pwd2_filtered = [p for p in pwd2_list if p.shape[0] == ref_len_pwd]

        if not pwd1_filtered or not pwd2_filtered:
            print("Warning: PwD vectors have inconsistent lengths or became empty after filtering for TICA. Skipping JS-TIC.")
            return np.nan

        data1 = np.ascontiguousarray(np.vstack(pwd1_filtered), dtype=np.float64)
        data2 = np.ascontiguousarray(np.vstack(pwd2_filtered), dtype=np.float64)
        
        # Re-check after potential filtering of pwd vectors
        if data2.shape[0] <= lagtime:
             print(f"Warning: Reference ensemble size for TICA after PwD processing ({data2.shape[0]}) "
                   f"is not greater than lagtime ({lagtime}). Skipping JS-TIC.")
             return np.nan

        print("Fitting TICA...")
        # Adjust n_components_requested if it's too large for the data
        # Max possible components is min(n_features, n_samples_after_lagging - 1)
        max_possible_components = min(data2.shape[1], data2.shape[0] - lagtime -1)
        if max_possible_components < 1:
            print(f"Warning: Not enough degrees of freedom for TICA (max_possible_components={max_possible_components}). Skipping JS-TIC.")
            return np.nan
            
        actual_n_components_to_fit = min(n_components_requested, max_possible_components)
        if actual_n_components_to_fit < n_components_requested:
            print(f"Warning: Requested {n_components_requested} TICA components, but data only supports {actual_n_components_to_fit}. Using {actual_n_components_to_fit}.")
        if actual_n_components_to_fit == 0 : # Should be caught by max_possible_components < 1
             print(f"Warning: Cannot fit any TICA components. Skipping JS-TIC.")
             return np.nan


        tica_estimator = TICA(dim=actual_n_components_to_fit, lagtime=lagtime).fit(data2)

        print("Transforming data...")
        proj1 = tica_estimator.transform(data1)
        proj2 = tica_estimator.transform(data2)

        # Get the actual number of components returned by TICA
        n_components_found = proj1.shape[1]
        if n_components_found == 0:
            print("Warning: TICA transformation resulted in 0 components. Skipping JS-TIC.")
            return np.nan
        if n_components_found < actual_n_components_to_fit:
             print(f"Warning: TICA fitted for {actual_n_components_to_fit} components, but transform yielded {n_components_found}.")


        js_divs = []
        # Loop up to the number of components actually found
        for i in range(n_components_found):
            comp1 = proj1[:, i]
            comp2 = proj2[:, i]
            combined_comp = np.concatenate((comp1, comp2))
            if combined_comp.size == 0:
                 js_divs.append(np.nan)
                 continue
            data_range = (np.min(combined_comp), np.max(combined_comp))
            hist1_tic, _ = get_histogram(comp1, n_bins=n_bins, data_range=data_range)
            hist2_tic, _ = get_histogram(comp2, n_bins=n_bins, data_range=data_range)
            js_divs.append(js_divergence(hist1_tic, hist2_tic))

        valid_js_divs = [d for d in js_divs if not np.isnan(d)]
        return np.mean(valid_js_divs) if valid_js_divs else np.nan
    except Exception as e:
        print(f"Error during JS-TIC calculation: {e}")
        import traceback
        traceback.print_exc()
        return np.nan

# --- Validity Metric (Unchanged) ---
def _check_steric_clash(coords: np.ndarray, threshold: float = CLASH_THRESHOLD_CA) -> bool:
    assert coords.ndim == 2 and coords.shape[1] == 3
    if coords.shape[0] < 2: return True
    dist_matrix = distance.squareform(distance.pdist(coords))
    min_dist = np.min(dist_matrix[np.triu_indices_from(dist_matrix, k=1)])
    return min_dist >= threshold

@timer
def calculate_validity(coords_list: List[np.ndarray], threshold: float = CLASH_THRESHOLD_CA) -> float:
    if not coords_list: return 0.0
    print(f"Calculating Validity (clash threshold = {threshold} Å)...")
    clash_free_count = 0
    for coords in tqdm(coords_list, desc="Checking Validity"):
        if _check_steric_clash(coords, threshold):
            clash_free_count += 1
    return clash_free_count / len(coords_list)

# --- TMscore / RMSD Calculation Helpers (Unchanged) ---
def calculate_rmsd_pair(coords1: np.ndarray, coords2: np.ndarray) -> float:
    if coords1.shape != coords2.shape: return np.inf
    if coords1.shape[0] == 0: return np.inf
    sup = SVDSuperimposer(); sup.set(coords1, coords2); sup.run()
    return sup.get_rms()

def calculate_tmscore_pair(pdb_path1: Path, pdb_path2: Path) -> float:
    tmalign_exec = get_base_dir("tmalign_executable")
    if not tmalign_exec.is_file(): return np.nan
    cmd = [str(tmalign_exec), str(pdb_path1), str(pdb_path2)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=90)
        if result.returncode != 0: return np.nan
        output = result.stdout; tm_score = np.nan
        for line in output.splitlines():
            if line.startswith("TM-score=") and "(if normalized by length of Chain_2" in line:
                try: tm_score = float(line.split()[1]); break
                except: continue
        return tm_score
    except Exception: return np.nan

# --- Ensemble TM/RMSD Metrics (Unchanged Logic) ---
@timer
def calculate_tm_ens(generated_pdb_dir: Path, reference_pdb_dir: Path,
                     gen_pdb_filenames: List[str], ref_pdb_filenames: List[str]) -> float:
    if not gen_pdb_filenames or not ref_pdb_filenames: return np.nan
    total_max_tm = 0; num_ref_valid = 0
    print(f"Calculating TM-ens ({len(gen_pdb_filenames)} generated vs {len(ref_pdb_filenames)} reference)...")
    for ref_fname in tqdm(ref_pdb_filenames, desc="Reference PDBs (TM-ens)"):
        ref_pdb_full_path = reference_pdb_dir / ref_fname
        max_tm_for_ref = 0.0; found_comparable = False
        for gen_fname in gen_pdb_filenames:
            gen_pdb_full_path = generated_pdb_dir / gen_fname
            tm = calculate_tmscore_pair(gen_pdb_full_path, ref_pdb_full_path)
            if not np.isnan(tm):
                found_comparable = True
                if tm > max_tm_for_ref: max_tm_for_ref = tm
        if found_comparable: total_max_tm += max_tm_for_ref; num_ref_valid += 1
    if num_ref_valid == 0: return np.nan
    return total_max_tm / num_ref_valid

@timer
def calculate_rmsd_ens(generated_coords_list: List[np.ndarray], reference_coords_list: List[np.ndarray]) -> float:
    if not generated_coords_list or not reference_coords_list: return np.nan
    total_min_rmsd = 0; num_ref_valid = 0
    print(f"Calculating RMSD-ens ({len(generated_coords_list)} generated vs {len(reference_coords_list)} reference)...")
    for ref_coords in tqdm(reference_coords_list, desc="Reference Coords (RMSD-ens)"):
        min_rmsd_for_ref = np.inf; found_comparable = False
        for gen_coords in generated_coords_list:
            rmsd = calculate_rmsd_pair(gen_coords, ref_coords)
            if not np.isinf(rmsd):
                found_comparable = True
                if rmsd < min_rmsd_for_ref: min_rmsd_for_ref = rmsd
        if found_comparable and not np.isinf(min_rmsd_for_ref):
             total_min_rmsd += min_rmsd_for_ref; num_ref_valid += 1
    if num_ref_valid == 0: return np.nan
    return total_min_rmsd / num_ref_valid