# eval_decoy_metrics.py
import argparse
import os
from pathlib import Path
import sys
import tempfile
import shutil # For copying files to temp dir if needed for TM-ens

import numpy as np
import pandas as pd

# --- Import Utilities ---
try:
    import eval_utils
except ImportError:
    script_dir = Path(__file__).resolve().parent
    sys.path.append(str(script_dir))
    try:
        import eval_utils
    except ImportError:
        print("CRITICAL Error: Could not import eval_utils.py. Make sure it's in the same directory or Python path.")
        exit(1)


# --- Argument Parsing ---
def get_argparser():
    parser = argparse.ArgumentParser(description="Evaluate protein decoy ensembles against a ground truth ensemble.")
    parser.add_argument("--protein_id", required=True, type=str,
                        help="Protein target ID (e.g., T1033).")
    parser.add_argument("--model_name", required=True, type=str,
                        help="Name of the model whose decoys are being evaluated (e.g., esmdiff, bioemu).")
    parser.add_argument("--native_filter", type=str, default="all", choices=["all", "near", "non"],
                        help="Filter model decoys based on TM-score to native structure ('all', 'near', 'non'). Default: 'all'.")
    parser.add_argument("--tmscore_threshold", type=float, default=0.5,
                        help="TM-score threshold for 'near'/'non' filtering. Default: 0.5.")
    # Corrected argument names here as per previous discussion
    parser.add_argument("--model_decoys_root_dir", type=str, required=True,
                        help="Path to the root directory for model decoys (e.g., /mnt/dna01/library2/caspdynamics).")
    parser.add_argument("--native_pdb_root_dir", type=str, required=True,
                        help="Path to the root directory containing native PDB structures (e.g., /mnt/rna01/zyh/prjs/caspdynamics/data/caspdynamics_natives).")
    parser.add_argument("--gt_decoys_root_dir", type=str, required=True,
                        help="Path to the root directory of your MD benchmark dataset (e.g., /mnt/rna01/zyh/data/selected_database2).")
    parser.add_argument("--tmalign_path", type=str, required=True,
                        help="Full path to the TMalign executable.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save the evaluation results CSV file.")
    parser.add_argument("--skip_tica", action="store_true", help="Skip JS-TIC calculation.")
    parser.add_argument("--skip_tmens", action="store_true", help="Skip TM-ens calculation (requires TMalign).")
    return parser

# --- Main Evaluation Logic ---
def main(args):
    print("--- Starting Decoy Ensemble Evaluation ---")
    print(f"Protein ID: {args.protein_id}")
    print(f"Model Name: {args.model_name}")
    print(f"Native Filter: {args.native_filter} (Threshold: {args.tmscore_threshold if args.native_filter != 'all' else 'N/A'})")
    print(f"Output Directory: {args.output_dir}")

    try:
        # Use the correct argument names when calling set_base_dirs
        eval_utils.set_base_dirs(
            model_decoys_root_dir=args.model_decoys_root_dir,
            native_pdb_root_dir=args.native_pdb_root_dir,
            gt_decoys_root_dir=args.gt_decoys_root_dir,
            tmalign_exec_path=args.tmalign_path
        )

        model_decoys_root = eval_utils.get_base_dir("model_decoys_root")
        gt_decoys_root = eval_utils.get_base_dir("gt_decoys_root")

        model_dir = model_decoys_root / f"{args.model_name}_decoys" / args.protein_id
        # The CSV file is directly under the model_decoys_root / <model_name>_decoys /
        model_csv = model_decoys_root / f"{args.model_name}_decoys" / f"{args.protein_id}_decoys.csv"
        gt_dir = gt_decoys_root / args.protein_id / "decoys_MD"

        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        output_csv_file = output_path / f"{args.protein_id}_{args.model_name}_{args.native_filter}_metrics.csv"

    except Exception as e:
        print(f"CRITICAL Error setting up paths or directories: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

    print("\n--- Loading Data ---")
    if not model_dir.is_dir():
        print(f"Error: Model decoy directory not found: {model_dir}")
        exit(1)
    if not gt_dir.is_dir():
        print(f"Error: Ground truth decoy directory not found: {gt_dir}")
        exit(1)

    # Load using the updated 2-pass loader
    model_coords_all, model_filenames_all = eval_utils.load_coords_from_dir(model_dir, recursive=False)
    gt_coords_all, gt_filenames_all = eval_utils.load_coords_from_dir(gt_dir, recursive=True)

    if not model_coords_all:
        print(f"Error: No model coordinates loaded from {model_dir} matching consensus length. Exiting.")
        exit(1)
    if not gt_coords_all:
        print(f"Error: No ground truth coordinates loaded from {gt_dir} matching consensus length. Exiting.")
        exit(1)

    model_coords_filtered = list(model_coords_all) # Make a copy
    model_filenames_filtered = list(model_filenames_all) # Make a copy

    # Filtering logic remains the same
    if args.native_filter != "all":
        print(f"\n--- Filtering Model Decoys ({args.native_filter}, Threshold={args.tmscore_threshold}) ---")
        if not model_csv.is_file():
            print(f"Error: Model TM-score CSV file not found: {model_csv}. Cannot perform filtering. Exiting.")
            exit(1)
        try:
            tmscore_df = pd.read_csv(model_csv)
            tmscore_df['filename_only'] = tmscore_df['decoy'].apply(lambda x: os.path.basename(x))
            tmscore_map = pd.Series(tmscore_df.tmscore.values, index=tmscore_df.filename_only).to_dict()

            temp_model_coords = []
            temp_model_filenames = []
            # model_filenames_all now contains names relative to model_dir (if recursive=False, just filename)
            for i, fname in enumerate(model_filenames_all):
                # Extract just the basename in case fname includes subdirs (shouldn't for model dir)
                fname_base = os.path.basename(fname)
                tm = tmscore_map.get(fname_base, None)
                if tm is None:
                    # print(f"Warning: TM-score not found for {fname_base} in CSV. Excluding.")
                    continue
                if args.native_filter == "near" and tm >= args.tmscore_threshold:
                    temp_model_coords.append(model_coords_all[i])
                    temp_model_filenames.append(model_filenames_all[i])
                elif args.native_filter == "non" and tm < args.tmscore_threshold:
                    temp_model_coords.append(model_coords_all[i])
                    temp_model_filenames.append(model_filenames_all[i])

            model_coords_filtered = temp_model_coords
            model_filenames_filtered = temp_model_filenames
            print(f"Filtered model ensemble size: {len(model_coords_filtered)} (from {len(model_coords_all)} originally loaded)")

            if not model_coords_filtered:
                print("Error: No model decoys remaining after filtering. Exiting.")
                exit(1)
        except Exception as e:
            print(f"Error during filtering: {e}")
            exit(1)
    else:
        print("\n--- No Model Filtering Applied ---")

    print(f"Model ensemble size for evaluation: {len(model_coords_filtered)}")
    print(f"Ground truth ensemble size for evaluation: {len(gt_coords_all)}")


    print("\n--- Calculating Metrics ---")
    results = {
        'protein_id': args.protein_id,
        'model_name': args.model_name,
        'native_filter': args.native_filter,
        'tmscore_threshold': args.tmscore_threshold if args.native_filter != 'all' else np.nan,
        'model_ensemble_size': len(model_coords_filtered),
        'gt_ensemble_size': len(gt_coords_all)
    }

    # Calculate metrics using the filtered model data and all GT data
    val, time = eval_utils.calculate_js_pwd(model_coords_filtered, gt_coords_all)
    results['JS_PwD'] = val; results['JS_PwD_time'] = time
    val, time = eval_utils.calculate_js_rg(model_coords_filtered, gt_coords_all)
    results['JS_Rg'] = val; results['JS_Rg_time'] = time

    if not args.skip_tica:
        val, time = eval_utils.calculate_js_tica(model_coords_filtered, gt_coords_all)
        results['JS_TIC'] = val; results['JS_TIC_time'] = time
    else:
        results['JS_TIC'], results['JS_TIC_time'] = np.nan, np.nan

    val, time = eval_utils.calculate_validity(model_coords_filtered)
    results['Validity_Model'] = val; results['Validity_Model_time'] = time

    val, time = eval_utils.calculate_rmsd_ens(model_coords_filtered, gt_coords_all)
    results['RMSD_ens'] = val; results['RMSD_ens_time'] = time

    if not args.skip_tmens:
        # TM-ens now uses the filtered list of model filenames and the full list of GT filenames
        val, time = eval_utils.calculate_tm_ens(model_dir, gt_dir, model_filenames_filtered, gt_filenames_all)
        results['TM_ens'] = val; results['TM_ens_time'] = time
    else:
        results['TM_ens'], results['TM_ens_time'] = np.nan, np.nan

    print("\n--- Saving Results ---")
    results_df = pd.DataFrame([results])
    try:
        results_df.to_csv(output_csv_file, index=False, float_format='%.6f')
        print(f"Results saved to: {output_csv_file}")
    except Exception as e:
        print(f"Error saving results to CSV: {e}")

    print("\n--- Evaluation Complete ---")

if __name__ == "__main__":
    parser = get_argparser()
    args = parser.parse_args()
    main(args)