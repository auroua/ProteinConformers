import os
import subprocess
import numpy as np
import argparse
from src.pdbench.utils.data_utils import extract_tm_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tm_exec_path", type=str)
    parser.add_argument("--native_pdb_path", type=str)
    parser.add_argument("--target_path", type=str)
    args = parser.parse_args()

    target_mapping = {id: os.path.join(args.target_path, id) for id in os.listdir(args.target_path) }

    total_tm_scores = []
    for f in target_mapping:
        cmd = [args.tm_exec_path, args.native_pdb_path, str(target_mapping[f])]
        print(str(target_mapping[f]))
        try:
            output = subprocess.check_output(cmd, text=True)
            tm_score = extract_tm_score(output)
            total_tm_scores.append(tm_score)
            print(f"{f}: {tm_score}")
        except subprocess.CalledProcessError as e:
            print(f"Failed to run TMscore for {f}: {e}")

    tm_score_data = np.array(total_tm_scores)
    print(tm_score_data)
    print(np.mean(tm_score_data), np.max(tm_score_data), np.min(tm_score_data), np.median(tm_score_data))