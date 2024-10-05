#!/bin/bash

#SBATCH --partition=single
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=20000mb
#SBATCH --time=08:00:00
#SBATCH --output=job_output_%j.log


#SBATCH --mail-user=zxoul27@student.uni-tuebingen.de
#SBATCH --mail-type=BEGIN,END,FAIL

# Set up the correct library paths
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

srun /pfs/data5/home/tu/tu_tu/tu_zxoul27/micromamba/envs/energy/bin/python sim_future_covariates.py

