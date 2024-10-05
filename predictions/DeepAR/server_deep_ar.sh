#!/bin/bash

#SBATCH --partition=gpu_4  
#SBATCH --gres=gpu:1
#SBATCH --nodes=1                     # Request 1 node
#SBATCH --cpus-per-task=8             # 8 CPU cores per task
#SBATCH --mem=16000mb                  # 32 GB of memory
#SBATCH --time=03:00:00               # Set the maximum runtime to 6 hours

#SBATCH --mail-user=zxoul27@student.uni-tuebingen.de  # Email notifications
#SBATCH --mail-type=BEGIN,END,FAIL    # Email on job start, end, and fail

# Set up the correct library paths
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

srun /pfs/data5/home/tu/tu_tu/tu_zxoul27/micromamba/envs/power/bin/python DeepAR_2.py

