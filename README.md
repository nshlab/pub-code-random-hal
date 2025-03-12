# Statistical Simulations in Julia
A template repository for setting up reproducible statistical simulations in Julia. `DrWatson.jl` is used to provide relative file path access (similar to `here` in R), run simulations faster and easier, and ensure the project is easily reproducible by others.

## Getting Started

When starting a project, be sure to write code and store data and experimental outputs in the correct location. Here is a detailed description of each directory:

- `notebooks` : Start your project by recording initial ideas and mini-experiments in Quarto notebooks here. Towards the end of the project, you can replace these with notebooks that generate tables and fancy plots to be included in a final manuscript. 
- `src` : Everything in this folder is saved as a "module" that is activated using the @quickactivate function in Julia. Once you are ready to define and run large-scale experiments, write functions that can run simulations with arbitrary sets of parameters here. Accessible via `srcdir()`
- `scripts` : This folder is used to store Julia scripts that produce plot and data output. These are the scripts that run various simulations with specific sets of parameters, potentially from the command line on a computing cluster. Accessible via `scriptsdir()`
- `data` : This folder stores both raw data and experimental outputs to be later analyzed here. Accessible via `datadir()`.
- `plots`: This folder stores plots, including those destined for a final manuscript  or presentation here. Accessible via `plotsdir()`.


## Cluster Computing

This repository is designed to ensure running code on the Harvard FASRC cluster is as simple as possible. Therefore, we include:

- `setting-up-fasrc.md` : A guide to setting up Julia on the FASRC cluster.
- `computational-practices.md` : Some tips and tricks for running computational experiments, including parallelized code in Julia.
- `sim-slurm.sh` : a SLURM script to run a Julia simulation on the cluster.

## Ready to Publish?

Delete the cluster computing guides and replace this README with a description of your research project. Then, make the repository public and share it with the world!
