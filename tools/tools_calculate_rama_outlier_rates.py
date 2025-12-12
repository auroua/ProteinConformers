# -*- coding: utf-8 -*-
"""
Created on Mon Jul 28 03:15:39 2025

@author: Yihang Zhou

Contact: yihangjoe@foxmail.com
         https://github.com/Y-H-Joe/

================================ description ==================================
Given a PDB list, output Ramachandran outlier rates.
Optional: if the input decoys.csv contains TM-score information, results can be grouped by TM-score and reported per group.
=================================== input =====================================

=================================== output ====================================

================================= parameters ==================================

=================================== example ===================================

=================================== warning ===================================
"""

import os
import sys
import csv
import glob
import random
import argparse
import tempfile
from tqdm import tqdm
import traceback
import pandas as pd
import pyrosetta
from pyrosetta.rosetta.core.scoring import Ramachandran

# ---------------- PyRosetta initialization ----------------
pyrosetta.init("-mute all")
rama = Ramachandran()

def _calc_rama_rate(pose):
    total, out = 0, 0
    for i in range(1, pose.total_residue() + 1):
        res = pose.residue(i)
        # Only evaluate protein residues; skip waters, ligands, nucleic acids, etc.
        if not res.is_protein():
            continue
        # Skip chain termini (N/C termini do not have complete ϕ/ψ)
        if res.is_lower_terminus() or res.is_upper_terminus():
            continue
        try:
            phi = pose.phi(i)
            psi = pose.psi(i)
        except RuntimeError:
            # In rare cases (e.g., missing backbone atoms) angles cannot be obtained; skip directly.
            continue
        if rama.phipsi_in_forbidden_rama(res.aa(), phi, psi):
            out += 1
        total += 1
    return 100.0 * out / total if total else None


def rama_outlier_rate(pdb_file: str):
    """
    If the input is a .cif, convert it to a temporary .pdb first, then compute; delete the temp file automatically after computation.
    """
    # ---------- Handle CIF ----------
    if pdb_file.lower().endswith(".cif"):
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".pdb")
        try:
            from Bio.PDB import MMCIFParser, PDBIO
            parser = MMCIFParser(QUIET=True)
            structure = parser.get_structure("struct", pdb_file)
            io = PDBIO()
            io.set_structure(structure)
            io.save(tmp_path)
            pose = pyrosetta.pose_from_pdb(tmp_path)
            return _calc_rama_rate(pose)
        except Exception:
            traceback.print_exc()
            return None
        finally:
            if os.path.exists(tmp_path):
                os.close(tmp_fd)
                os.remove(tmp_path)
    # ---------- Handle PDB ----------
    else:
        try:
            pose = pyrosetta.pose_from_pdb(pdb_file)
        except RuntimeError:
            traceback.print_exc()
            return None
        return _calc_rama_rate(pose)

def normalize_cols(df):
    """Return a mapping dict of normalized column names to original names: norm_name -> real_name."""
    return {c.strip().lower(): c for c in df.columns}

def write_rows(output_dp, rows, header_written_flag):
    """Append rows to CSV; write header depending on header_written_flag. Return the updated header flag."""
    if not rows:
        return header_written_flag
    fieldnames = ["decoy", "casp_id", "rama_outlier_rate", "tmscore"]
    mode = "a"
    with open(output_dp, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not header_written_flag:
            writer.writeheader()
            header_written_flag = True
        writer.writerows(rows)
    return header_written_flag


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute Rama outlier rates per decoy, optionally guided by {casp_id}_decoys.csv.")
    parser.add_argument("method_dir", help="Root directory: contains multiple casp_id subfolders; each subfolder contains PDB/CIF files.")
    parser.add_argument("output_dp", help="Output CSV (one structure per row).")
    parser.add_argument("--decoy_csv_dir", default=None,
                        help="If provided, drive by {casp_id}_decoys.csv in this directory; required column: decoy; if tmscore exists, it will be output as well.")
    parser.add_argument("--max_per_casp", type=int, default=1000,
                        help="When decoy-csv is not provided, process at most this many structures per casp by random sampling (default 1000).")
    args = parser.parse_args()

    method_dir = args.method_dir
    output_dp  = args.output_dp
    decoy_csv_dir = args.decoy_csv_dir
    max_per_casp = max(1, int(args.max_per_casp))

    os.makedirs(os.path.dirname(output_dp) or ".", exist_ok=True)
    log_fp = output_dp + ".processed_ids.txt"

    # ---------- Completed log ----------
    if os.path.exists(log_fp):
        with open(log_fp) as f:
            processed_ids = {line.strip() for line in f if line.strip()}
    else:
        processed_ids = set()

    # Whether the header has already been written
    header_written = os.path.exists(output_dp) and os.path.getsize(output_dp) > 0

    # List of casp directories
    casp_ids = [d for d in os.listdir(method_dir) if os.path.isdir(os.path.join(method_dir, d))]
    casp_ids.sort()

    for casp_id in tqdm(casp_ids, desc="casp folders"):
        casp_path = os.path.join(method_dir, casp_id)

        rows_to_write = []

        # ============ Mode A: driven by {casp_id}_decoys.csv ============
        decoy_csv_path = None
        if decoy_csv_dir is not None:
            decoy_csv_path = os.path.join(decoy_csv_dir, f"{casp_id}_decoys.csv")
            if not os.path.isfile(decoy_csv_path):
                decoy_csv_path = None  # Not found; fall back

        if decoy_csv_path:
            try:
                df_decoy = pd.read_csv(decoy_csv_path)
            except Exception as e:
                print(f"[WARN] Failed to read {decoy_csv_path}: {e}. Falling back to directory-scan mode.")
                df_decoy = None

            if df_decoy is not None and df_decoy.shape[1] > 0:
                c = normalize_cols(df_decoy)
                if "decoy" not in c:
                    print(f"[WARN] {decoy_csv_path} is missing the 'decoy' column. Falling back to directory-scan mode.")
                else:
                    decoy_series = df_decoy[c["decoy"]].astype(str).str.strip()
                    # If the CSV includes a casp_id column, filter once to avoid mixing targets.
                    if "casp_id" in c:
                        df_decoy = df_decoy[df_decoy[c["casp_id"]].astype(str) == str(casp_id)]
                        decoy_series = df_decoy[c["decoy"]].astype(str).str.strip()

                    # Optional tmscore
                    tms = None
                    if "tmscore" in c:
                        tms = pd.to_numeric(df_decoy[c["tmscore"]], errors="coerce")
                    else:
                        tms = pd.Series([None] * len(decoy_series))

                    # Insert immediately after reading decoy_series and tms
                    indices = list(range(len(decoy_series)))
                    if len(indices) > max_per_casp:
                        indices = random.sample(indices, max_per_casp)
                    # Reorder to the sampled subset and reset indices to keep tqdm(total=...) correct
                    decoy_series = decoy_series.iloc[indices].reset_index(drop=True)
                    tms = tms.iloc[indices].reset_index(drop=True)

                    for decoy_path, tms_val in tqdm(zip(decoy_series, tms),
                                                    total=len(decoy_series),
                                                    desc=f"{casp_id} decoys",
                                                    leave=False):
                        key = decoy_path  # Logging granularity: decoy path
                        if key in processed_ids:
                            continue
                        if not os.path.exists(decoy_path):
                            # Some decoys may not exist on the local machine; log and skip
                            processed_ids.add(key)
                            with open(log_fp, "a") as f:
                                f.write(key + "\n")
                            continue
                        rate = rama_outlier_rate(decoy_path)
                        if rate is None:
                            # Log failures as well to avoid endless retries
                            processed_ids.add(key)
                            with open(log_fp, "a") as f:
                                f.write(key + "\n")
                            continue
                        rows_to_write.append({
                            "decoy": decoy_path,
                            "casp_id": casp_id,
                            "rama_outlier_rate": rate,
                            "tmscore": None if pd.isna(tms_val) else float(tms_val),
                        })
                        processed_ids.add(key)
                        with open(log_fp, "a") as f:
                            f.write(key + "\n")

                    header_written = write_rows(output_dp, rows_to_write, header_written)
                    continue  # Move on to next casp_id

        # ============ Mode B: scan directory (fallback) ============
        pdb_files = [f for f in os.listdir(casp_path)
                     if (f.lower().endswith(".pdb") or f.lower().endswith(".cif"))]
        if not pdb_files:
            continue

        # Sampling (or use all if file count is below the limit)
        if len(pdb_files) > max_per_casp:
            pdb_subset = random.sample(pdb_files, max_per_casp)
        else:
            pdb_subset = pdb_files

        for f in tqdm(pdb_subset, desc=f"{casp_id} decoys", leave=False):
            decoy_path = os.path.join(casp_path, f)
            key = f"{casp_id}::{f}"  # In fallback mode, keep the original log granularity (casp_id-level to avoid huge logs)
            if key in processed_ids:
                continue
            if not os.path.exists(decoy_path):
                processed_ids.add(key)
                with open(log_fp, "a") as lf:
                    lf.write(key + "\n")
                continue
            rate = rama_outlier_rate(decoy_path)
            if rate is None:
                processed_ids.add(key)
                with open(log_fp, "a") as lf:
                    lf.write(key + "\n")
                continue
            rows_to_write.append({
                "decoy": decoy_path,
                "casp_id": casp_id,
                "rama_outlier_rate": rate,
                "tmscore": None,  # No tmscore in fallback mode
            })
            processed_ids.add(key)
            with open(log_fp, "a") as lf:
                lf.write(key + "\n")
        header_written = write_rows(output_dp, rows_to_write, header_written)
    
    if False: # summarize and calculate mean
        rama_dir = r"D:\Projects\ProteinConformers\data\rama_outlier_rates"
        results = {}
        for file in os.listdir(rama_dir):
            if os.path.isfile(os.path.join(rama_dir,file)) and file.endswith('.csv'):
                df = pd.read_csv(os.path.join(rama_dir,file))
                results[file] = df['rama_outlier_rate'].mean()
