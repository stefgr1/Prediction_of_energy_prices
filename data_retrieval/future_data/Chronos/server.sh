#!/bin/bash

#SBATCH --partition=single     # Ensure this partition is valid
#SBATCH --nodes=1                   # Request a single node
#SBATCH --cpus-per-task=25          # Specify the number of CPU cores for the task
#SBATCH --mem=50GB              # Set memory for the job
#SBATCH --time=12:00:00             # Set maximum runtime
#SBATCH --output=job_output_%j.out
#SBATCH --error=job_error_%j.err
#SBATCH --job-name=TTF_gas
#SBATCH --mail-user=zxoul27@student.uni-tuebingen.de
#SBATCH --mail-type=BEGIN,END,FAIL

# Set up the correct library paths
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Run your Python script
srun /pfs/data5/home/tu/tu_tu/tu_zxoul27/micromamba/envs/power/bin/python chronos_cpu_par.py

