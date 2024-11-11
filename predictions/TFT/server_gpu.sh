#!/bin/bash

#SBATCH --job-name=tftnolag
#SBATCH --partition=gpu_8
#SBATCH --gres=gpu:2
#SBATCH --nodes=1                     # Request 1 node
#SBATCH --cpus-per-task=6             # 6 CPU cores per task
#SBATCH --mem=10GB                    # 12 GB of memory
#SBATCH --time=24:00:00             
#SBATCH --output=job_output_%j.out
#SBATCH --error=job_error_%j.err    
#SBATCH --mail-user=zxoul27@student.uni-tuebingen.de  # Email notifications
#SBATCH --mail-type=BEGIN,END,FAIL    # Email on job start, end, and fail

# Set up the correct library paths
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

srun /pfs/data5/home/tu/tu_tu/tu_zxoul27/micromamba/envs/power/bin/python TFT_model.py