#!/bin/bash
#SBATCH --job-name=bid4
#SBATCH --nodes=1
#SBATCH --array=1-10
#SBATCH --ntasks-per-node=5
#SBATCH --time=5-1:00:00
#SBATCH --mem=20gb
#SBATCH --output=bid4.out
#SBATCH --error=bid4.err

source /home/software/conda/miniconda3/etc/profile.d/conda.sh
conda activate python311

python run_sim_out.py $SLURM_ARRAY_TASK_ID
