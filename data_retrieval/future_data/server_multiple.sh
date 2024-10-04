#!/bin/bash

#SBATCH --partition=multiple               # Assuming a multi-node partition is available
#SBATCH --nodes=2                       # Request 2 nodes
#SBATCH --ntasks=8                      # Run 8 tasks (total across all nodes)
#SBATCH --ntasks-per-node=4             # Run 4 tasks per node
#SBATCH --cpus-per-task=8               # 4 CPU cores per task
#SBATCH --mem=64000mb                   # 64GB of memory per node
#SBATCH --time=06:00:00                 # Set the maximum run time
#SBATCH --output=job_output_%j.log      # Output log file for each job

#SBATCH --mail-user=zxoul27@student.uni-tuebingen.de
#SBATCH --mail-type=BEGIN,END,FAIL

/pfs/work7/workspace/scratch/tu_zxoul27-master_thesis/miniconda3/envs/master_thesis/bin/python sim_cov_multiple.py
