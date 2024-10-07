#!/bin/bash

#SBATCH --partition=multiple       # Assuming a multi-node partition is available
#SBATCH --nodes=1                      # Request 1 nodes
#SBATCH --ntasks=12                    # 12 tasks in total, evenly distributed across the nodes
#SBATCH --ntasks-per-node=6            # Distribute tasks across nodes
#SBATCH --cpus-per-task=6              # 4 CPU cores per task
#SBATCH --mem=16000mb                  # 32GB of memory per node
#SBATCH --time=10:00:00                # Set the maximum run time
#SBATCH --output=job_output_%j.log     # Output log file for each job

#SBATCH --mail-user=zxoul27@student.uni-tuebingen.de
#SBATCH --mail-type=BEGIN,END,FAIL

# Set up the correct library paths
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

srun /pfs/data5/home/tu/tu_tu/tu_zxoul27/micromamba/envs/power/bin/python XGBoost_cluster.py

