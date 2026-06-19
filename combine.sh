#!/bin/bash
#SBATCH -J combine
#SBATCH -o combine%j.out
#SBATCH -e combine%j.err
#SBATCH -p hejazi
#SBATCH -t 0-00:30
#SBATCH -c 1
#SBATCH --mem=12G

# Put what you want to do with singularity below
# The line below runs a Julia script called "juliatest.jl"
srun -c ${SLURM_CPUS_PER_TASK} singularity exec ../julia_latest.sif julia 'scripts/99_combiner.jl'
#singularity exec julia_latest.sif julia --threads 1 'juliatest.jl'


