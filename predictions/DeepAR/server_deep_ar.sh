#!/bin/bash

#SBATCH --partition=gpu_8  
#SBATCH --gres=gpu:2
#SBATCH --nodes=1                     # Request 1 node
#SBATCH --cpus-per-task=8             # 8 CPU cores per task
#SBATCH --mem=16000mb                  # 32 GB of memory
#SBATCH --time=03:00:00                 
#SBATCH --mail-user=zxoul27@student.uni-tuebingen.de  # Email notifications
#SBATCH --mail-type=BEGIN,END,FAIL    # Email on job start, end, and fail

# Set up the correct library paths
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

srun /pfs/data5/home/tu/tu_tu/tu_zxoul27/micromamba/envs/power/bin/python DeepAR_2.py

