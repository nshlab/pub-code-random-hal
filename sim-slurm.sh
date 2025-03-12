#!/bin/bash
#SBATCH -J sim_name
#SBATCH -o sim_name%j.out
#SBATCH -e sim_name%j.err
#SBATCH -p hejazi
#SBATCH -t 0-6:00:00
#SBATCH -c 41
#SBATCH --mem-per-cpu=10000

# Put what you want to do with singularity below
srun -c $SLURM_CPUS_PER_TASK singularity exec julia_latest.sif julia --threads 41 'path/to/script.jl' 'arg1' 'arg2'
