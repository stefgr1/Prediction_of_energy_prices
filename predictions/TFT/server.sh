#!/bin/bash

#SBATCH --job-name=tft_cpu
#SBATCH --partition=single
#SBATCH --nodes=1                     # Request 2 nodes
#SBATCH --cpus-per-task=40            # 10 CPU cores per task
#SBATCH --mem=16GB                    # Memory per node
#SBATCH --time=24:00:00               # Maximum runtime of 24 hours
#SBATCH --output=job_output_%j.out
#SBATCH --error=job_error_%j.err
#SBATCH --job-name=TFT_nol  # Set the job name
#SBATCH --mail-user=zxoul27@student.uni-tuebingen.de  # Email notifications
#SBATCH --mail-type=BEGIN,END,FAIL    # Email on job start, end, and fail

# Set up the correct library paths
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

srun /pfs/data5/home/tu/tu_tu/tu_zxoul27/micromamba/envs/power/bin/python TFT_model.py

