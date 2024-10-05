#!/bin/bash

#SBATCH --partition=gpu_4         # Request the dev_gpu_4 partition
#SBATCH --nodes=1                     # Request 1 node
#SBATCH --cpus-per-task=6             # 8 CPU cores per task
#SBATCH --mem=8000mb                 # 32 GB of memory
#SBATCH --time=05:00:00               # Set the maximum runtime to 6 hours
#SBATCH --gres=gpu:1                  # Request 1 GPU

#SBATCH --mail-user=zxoul27@student.uni-tuebingen.de  # Email notifications
#SBATCH --mail-type=BEGIN,END,FAIL    # Email on job start, end, and fail

# Set up the correct library paths
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

srun /pfs/data5/home/tu/tu_tu/tu_zxoul27/micromamba/envs/energy/bin/python TFT_model_2.py

