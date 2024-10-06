#!/bin/bash

# SBATCH --partition=single
# SBATCH --nodes=1
# SBATCH --cpus-per-task=8
# SBATCH --mem=10000mb
# SBATCH --time=06:00:00

# SBATCH --mail-user=zxoul27@student.uni-tuebingen.de
# SBATCH --mail-type=BEGIN,END,FAIL

source / pfs/work7/workspace/scratch/tu_zxoul27 - \
    master_thesis/miniconda3/etc/profile.d/conda.sh
conda activate master_thesis

python sim_future_covariates.py
