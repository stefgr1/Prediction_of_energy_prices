#!/bin/bash

#SBATCH --partition=multiple       # Assuming a multi-node partition is available
#SBATCH --nodes=2                      # Request 2 nodes
#SBATCH --ntasks=12                    # 12 tasks in total, evenly distributed across the nodes
#SBATCH --ntasks-per-node=6            # Distribute tasks across nodes
#SBATCH --cpus-per-task=6              # 4 CPU cores per task
#SBATCH --mem=90000mb                  # 32GB of memory per node
#SBATCH --time=15:00:00                # Set the maximum run time
#SBATCH --output=job_output_%j.log     # Output log file for each job

#SBATCH --mail-user=zxoul27@student.uni-tuebingen.de
#SBATCH --mail-type=BEGIN,END,FAIL

# Set up the correct library paths
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Use srun to ensure proper task allocation
srun /pfs/data5/home/tu/tu_tu/tu_zxoul27/micromamba/envs/energy/bin/python sim_cov_multiple.py
