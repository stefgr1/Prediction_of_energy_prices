#!/bin/bash

#SBATCH --job-name=final_Sim
#SBATCH --partition=dev_gpu_4
#SBATCH --gres=gpu:2
#SBATCH --nodes=1                     # Request 1 node
#SBATCH --cpus-per-task=6             # 6 CPU cores per task
#SBATCH --mem=10GB                    # 12 GB of memory
#SBATCH --time=00:30:00             
#SBATCH --output=job_output_%j.out
#SBATCH --error=job_error_%j.err    
#SBATCH --mail-user=zxoul27@student.uni-tuebingen.de  # Email notifications
#SBATCH --mail-type=BEGIN,END,FAIL    # Email on job start, end, and fail

# Set up the correct library paths
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

srun /pfs/data5/home/tu/tu_tu/tu_zxoul27/micromamba/envs/power/bin/python tft_final_sim.py