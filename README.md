# ProteinConformers

Official repository for **ProteinConformers: Benchmark Dataset for Simulating Protein Conformational Landscape Diversity and Plausibility**.

## Overview
ProteinConformers provides data loaders, sampling pipelines, and evaluation utilities for benchmarking generative models of protein conformations. The repository currently includes reference implementations for BioEmu- and ESMdiff-based samplers and a suite of downstream metrics covering free-energy estimation, population coverage, and structural plausibility.

## Environment Setup
To generate decoy structures, users must install and correctly configure the environments for AlphaFlow, BioEmu, ESMdiff, AFsample2, and AlphaFold3. This repository includes turnkey sampling pipelines for BioEmu and ESMdiff only; decoys produced by AlphaFlow, AFsample2, and AlphaFold3 must be generated externally and then provided to this pipeline.

### Python environment (uv)
The project is managed with [uv](https://docs.astral.sh/uv/). Install uv if it is not already available:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Create the base environment (Python 3.10 or 3.11):
```bash
uv sync
```

The sync step resolves all core dependencies defined in `pyproject.toml` and produces a `.venv` directory in the project root. Use `uv run` to execute repository commands inside this environment, for example:
```bash
uv run python tools/tools_generate_conformations.py --help
```

### Optional: BioEmu ColabFold backend
BioEmu relies on a patched ColabFold installation for structure refinement. The following steps create the auxiliary environment and apply the required modifications:

```bash
* conda create -n colabfold_env python=3.10
* conda activate colabfold_env
* pip install uv
* export VENV_FOLDER=/mnt/rna01/chenw/anaconda3/envs/colabfold_env
* uv pip install --python ${VENV_FOLDER}/bin/python 'colabfold[alphafold-minus-jax]==1.5.4' 
* uv pip install --python ${VENV_FOLDER}/bin/python --force-reinstall "jax[cuda12]"==0.4.35 "numpy==1.26.4"
* export SITE_PACKAGES_DIR=${VENV_FOLDER}/lib/python3.10/site-packages
* patch ${SITE_PACKAGES_DIR}/alphafold/model/modules.py ${SCRIPT_DIR}/modules.patch 
* patch ${SITE_PACKAGES_DIR}/colabfold/batch.py ${SCRIPT_DIR}/batch.patch
* touch ${VENV_FOLDER}/.COLABFOLD_PATCHED
* The BIOEMU_COLABFOLD_DIR is `/mnt/rna01/chenw/anaconda3/envs/colabfold_env`
*  vi /mnt/rna01/chenw/WorkSpace_Bio/bioemu/src/bioemu/get_embeds.py, change the line of code `return subprocess.run(cmd, env=colabfold_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)` to `return subprocess.run(['conda', "run", "-n", "colabfold_env", *cmd], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)`
* pip install esm==3.0.4
* pip install -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple tokenizers
* pip install -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple transformers
```

Set the following environment variables before invoking the sampler:
```bash
export BIOEMU_COLABFOLD_DIR=/mnt/rna01/chenw/anaconda3/envs/colabfold_env
export CUDA_HOME=/mnt/apps/cuda_12.1.0
```

### Optional: ESMdiff environment
The ESMdiff baseline requires additional model checkpoints and configuration files. Please consult the documentation located in `configs/esmdiff` for detailed installation and usage instructions.

## Usage

### Sampling protein conformations

**BioEmu sampler**
```bash
uv run python tools/tools_generate_conformations.py \
    --fasta_file_path benchmark_seqs.fasta \
    --sampler_type bioemu \
    --sample_size 3000 \
    --save_path ./bioemu
```

**ESMdiff sampler**
```bash
uv run python tools/tools_generate_conformations.py \
    --fasta_file_path benchmark_seqs.fasta \
    --sampler_type esmdiff \
    --sample_size 3000 \
    --save_path ./esmdiff \
    --ckpt_path /mnt/rna01/chenw/WorkSpace_Bio/esmdiff/data/ckpt/release_v0.pt \
    --sample_mode ddpm \
    --sample_steps 1000 \
    --model_config_path ./configs/esmdiff/experiment/mdlm.yaml
```

### Evaluation utilities

#### Comprehensive Decoy Ensemble Evaluation

The `src/eval` module provides a comprehensive pipeline for evaluating the quality and similarity of protein decoy ensembles against ground truth conformational ensembles (typically from MD simulations). The evaluation includes multiple metrics inspired by the ESMDiff paper and common structural bioinformatics practices:

**Available Metrics:**
- **Jensen-Shannon Divergence (JS-Div):**
  - `JS-PwD`: Based on C-alpha pairwise distance distributions
  - `JS-Rg`: Based on Radius of Gyration distributions  
  - `JS-TIC`: Based on Time-lagged Independent Components (derived from pairwise distances)
- **Ensemble Coverage:**
  - `RMSD-ens`: Average minimum C-alpha RMSD of GT structures to the generated ensemble
  - `TM-ens`: Average maximum TM-score of GT structures to the generated ensemble
- **Structural Validity:**
  - `Validity_Model`: Fraction of clash-free structures in the generated ensemble

**Additional Dependencies for Evaluation:**
```bash
# Install evaluation-specific dependencies
pip install biopython deeptime

# Download and compile TMalign (required for TM-ens metric)
# Get TMalign from: https://zhanggroup.org/TM-align/
```

**Running Evaluation:**
```bash
# Basic evaluation example
uv run python src/eval/eval_decoy_metrics.py \
    --protein_id T1033 \
    --model_name esmdiff \
    --native_filter all \
    --model_decoys_root_dir /path/to/model/decoys \
    --native_pdb_root_dir /path/to/native/pdbs \
    --gt_decoys_root_dir /path/to/ground/truth \
    --tmalign_path /path/to/TMalign_cpp \
    --output_dir /path/to/output

# Options:
# --native_filter: Filter decoys by TM-score to native (all/near/non)
# --tmscore_threshold: TM-score threshold for filtering (default: 0.5)
# --skip_tica: Skip JS-TIC calculation (time-consuming)
# --skip_tmens: Skip TM-ens calculation (requires TMalign)
```

For detailed documentation and examples, see `src/eval/readme_eval.md`.

#### Other Evaluation Tools

1. Compute the free-energy landscape:

```bash
uv run python tools/tools_calculate_free_energy_landscape.py
```

2. Compute the energy overlap
```bash
uv run python tools/tools_calculate_fel_overlpas.py --generated-file-path /mnt/dna01/library2/caspdynamics/generated_data 
```

3. Compute population coverage scores:

```bash
# Protein Conformational Plausibility Score (PCPS)
uv run python tools/tools_pcps.py

# Per-protein PCPM statistics
uv run python tools/tools_pcpm_individual.py

# Aggregate PCPM distribution
uv run python tools/tools_pcpm_distribution.py
```


## Citation
If you use ProteinConformers in your research, please cite:
```bibtex
@inproceedings{ProteinConformers,
  author    = {Yihang Zhou, Chen Wei, Matthew M. Sun, Jin Song, Yang Li, Lin Wang and Yang Zhang},
  title     = {ProteinConformers: Benchmark Dataset for Simulating Protein Conformational Landscape Diversity and Plausibility},
  booktitle = {Proceedings of the 39th Conference on Neural Information Processing Systems (NeurIPS 2025)},
  year      = {2025},
  note      = {Poster},
  url       = {https://neurips.cc/virtual/2025/poster/121755},
  doi       = {},
  publisher = {}
}
```
