#!/bin/bash
#SBATCH -J large_randomhal
#SBATCH -o large_randomhal%j.out
#SBATCH -e large_randomhal%j.err
#SBATCH -p hsph
#SBATCH -t 0-48:00:00
#SBATCH -c 40
#SBATCH --mem=128GB

# Put what you want to do with singularity below
srun -c $SLURM_CPUS_PER_TASK singularity exec ../julia_latest.sif julia --threads 40 'scripts/2_large_randomhal.jl'
