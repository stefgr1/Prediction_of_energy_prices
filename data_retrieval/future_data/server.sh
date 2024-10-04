#!/bin/bash

#SBATCH --partition=single
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=20000mb
#SBATCH --time=08:00:00
#SBATCH --output=job_output_%j.log


#SBATCH --mail-user=zxoul27@student.uni-tuebingen.de
#SBATCH --mail-type=BEGIN,END,FAIL

/pfs/work7/workspace/scratch/tu_zxoul27-master_thesis/miniconda3/envs/master_thesis/bin/python sim_future_covariates.py

