#!/bin/bash

#SBATCH --partition=gpu_8
#SBATCH --gres=gpu:2
#SBATCH --nodes=1                     # Request 1 node
#SBATCH --cpus-per-task=6             # 6 CPU cores per task
#SBATCH --mem=8000mb                  # 12 GB of memory
#SBATCH --time=05:00:00                 
#SBATCH --job-name=Deep_AR_52  # Set the job name
#SBATCH --mail-user=zxoul27@student.uni-tuebingen.de  # Email notifications
#SBATCH --mail-type=BEGIN,END,FAIL    # Email on job start, end, and fail

# Set up the correct library paths
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

srun /pfs/data5/home/tu/tu_tu/tu_zxoul27/micromamba/envs/power/bin/python DeepAR_2.py

