#!/bin/bash

#SBATCH --partition=dev_gpu_4        # Ensure this partition is valid
#SBATCH --nodes=1                   # Request a single node
#SBATCH --cpus-per-task=6          # Specify the number of CPU cores for the task
#SBATCH --mem=8000mb               # Set memory for the job
#SBATCH --time=00:30:00             # Set maximum runtime
#SBATCH --output=job_output_%j.out
#SBATCH --error=job_error_%j.err
#SBATCH --gres=gpu:2                 # Request 1 GPU
#SBATCH --job-name=XG_Boost
#SBATCH --mail-user=zxoul27@student.uni-tuebingen.de
#SBATCH --mail-type=BEGIN,END,FAIL

# Set up the correct library paths
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Run your Python script
srun /pfs/data5/home/tu/tu_tu/tu_zxoul27/micromamba/envs/power/bin/python XGBoost_cluster_2.py
