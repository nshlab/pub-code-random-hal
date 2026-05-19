#!/bin/bash
#SBATCH --job-name=randomhal
#SBATCH --account=nhejazi_lab
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --array=1-800

# Put what you want to do with singularity below
srun -c $SLURM_CPUS_PER_TASK singularity exec julia_latest.sif julia 'scripts/0_orchestrator.jl'
