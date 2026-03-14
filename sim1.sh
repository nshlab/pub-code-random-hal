#!/bin/bash
#SBATCH -J small_comparison
#SBATCH -o small_comparison%j.out
#SBATCH -e small_comparison%j.err
#SBATCH -p hsph
#SBATCH -t 0-48:00:00
#SBATCH -c 40
#SBATCH --mem=128GB

# Put what you want to do with singularity below
srun -c $SLURM_CPUS_PER_TASK singularity exec ../julia_latest.sif julia --threads 40 'scripts/1_small_comparison.jl'
