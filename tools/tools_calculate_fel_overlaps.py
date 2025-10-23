import os
import pickle
import numpy as np
import argparse
from typing import List, Dict

def overlap_metrics(pc_array, method_array, threshold=40.0):
    """
    Compute overlap between low-energy regions (< threshold kJ/mol) of two 2D arrays.

    Parameters
    ----------
    pc_array : np.ndarray
        Ground-truth energy grid (2D).
    method_array : np.ndarray
        Method energy grid (2D). Must be the same shape as pc_array.
    threshold : float
        Energy cutoff for "low-energy" region.

    Returns
    -------
    intersection : float
        Number of grid points in the intersection A∩B of low-energy masks.
    coverage : float
        |A∩B| / |A| (how much of GT low-energy region A is covered by method B).
    jaccard : float
        |A∩B| / |A∪B| (IoU).
    """
    # Ensure same shape
    if pc_array.shape != method_array.shape:
        raise ValueError("Arrays must have same shape")
    # Masks for low-energy (< threshold)
    mask_A = pc_array < threshold
    mask_B = method_array < threshold

    # Count points
    intersection = np.count_nonzero(mask_A & mask_B) * 1.0
    A_area = np.count_nonzero(mask_A)
    B_area = np.count_nonzero(mask_B)
    union = A_area + B_area - intersection
    # print(intersection, union)
    # Compute metrics (with safe handling of zero-count cases)
    coverage = intersection / A_area if A_area > 0 else 0.0
    jaccard = intersection / union if union > 0 else 1.0

    return intersection, coverage, jaccard


def build_argparser() -> argparse.ArgumentParser:
    """CLI flags."""
    p = argparse.ArgumentParser(description="Overlap metrics for FEL low-energy regions.")
    p.add_argument(
        "--generated-file-path",
        default="/mnt/dna01/library2/caspdynamics/generated_data/",
        help="Directory containing pickled FEL mappings."
    )
    p.add_argument(
        "--methods",
        nargs="+",
        default=["alphaflow_md", "alphaflow_pdb", "esmflow_md", "esmflow_pdb", "bioemu", "esmdiff", "af3", "afsample2"],
        help="Methods to summarize; only present keys will be reported."
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=5.0,
        help="Low-energy threshold (kJ/mol) used for overlap."
    )
    return p


def main():
    args = build_argparser().parse_args()

    keys_list: List[str] = list(args.methods)

    method_overlap_intersection = {}
    method_overlap_coverage = {}
    method_overlap_jaccard = {}

    for f in os.listdir(args.generated_file_path):
        pkl_obj_file_path = os.path.join(args.generated_file_path, f)
        with open(pkl_obj_file_path, 'rb') as f:
            method_energy_landscape_mapping = pickle.load(f)
        method_energy_landscape_mapping.pop("native")

        _, _, gt_energy = method_energy_landscape_mapping.pop("ProteinConformers")

        for k, v in method_energy_landscape_mapping.items():
            _, _, method_energy = method_energy_landscape_mapping[k]
            intersection, coverage, jaccard = overlap_metrics(gt_energy, method_energy, threshold=args.threshold)
            if k in method_overlap_intersection:
                method_overlap_intersection[k].append(intersection)
                method_overlap_coverage[k].append(coverage)
                method_overlap_jaccard[k].append(jaccard)
            else:
                method_overlap_intersection[k] = [intersection]
                method_overlap_coverage[k] = [coverage]
                method_overlap_jaccard[k] = [jaccard]

    # Summaries
    def safe_mean(arr: List[float]) -> float:
        return float(np.mean(arr)) if arr else float("nan")

    # Print in the user-specified order; report only methods we actually saw
    print("\n=== Mean Intersection (grid points) ===")
    for k in keys_list:
        if k in method_overlap_intersection:
            print(f"{k:12s} {safe_mean(method_overlap_intersection[k]):.3f}")
        else:
            print(f"{k:12s} n/a")

    print("\n=== Mean Coverage (|A∩B|/|A|) ===")
    for k in keys_list:
        if k in method_overlap_coverage:
            print(f"{k:12s} {safe_mean(method_overlap_coverage[k]):.4f}")
        else:
            print(f"{k:12s} n/a")

    print("\n=== Mean Jaccard (IoU) ===")
    for k in keys_list:
        if k in method_overlap_jaccard:
            print(f"{k:12s} {safe_mean(method_overlap_jaccard[k]):.4f}")
        else:
            print(f"{k:12s} n/a")

    print("\nDone.")


if __name__ == "__main__":
    main()