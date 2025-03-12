# Setting up Julia on FASRC

This document instructs NSH lab members on how to get started running numerical simulations in Julia on the Harvard FASRC Computing Cluster. This tutorial assumes that you have already set up an account on FASRC and are able to log in via the command line. If not, follow the [Quickstart Guide](https://docs.rc.fas.harvard.edu/kb/quickstart-guide/). 

Here is a summary of all of the commands referenced in this guide:

- `singularity pull docker://julia` : get latest official version of Julia
- `salloc -p test --mem 1G -t 0-06:00` : enter an interactive session to update Julia packages
- `singularity shell julia_latest.sif` : enter the Singularity container of the latest version of Julia. Once in, type `julia` to start up a Julia REPL to manage packages.

## 0: What not to do

Firstly, Harvard's documentation will encourage you to use Julia through a Jupyter Notebook. If your goal is to run a compute-intensive job, **it is not recommended to do this**. Jupyyer Notebooks are not well-suited for running large jobs and are likely to kill any long-running process unexpectedly. 

Secondly, you should not install Julia on the cluster or attempt to run any computationally-intensive code in the Julia REPL, even in Singularity. Using the Julia REPL on the command line will run code on the login node, which is not intended for computationally-intensive tasks. 

Instead, you should write your Julia code as a script in a `.jl` file and submit it as a job to the cluster. Before doing this, however, you will need to set up a container on the cluster through which the job can be run.


## 1: Set up a Singularity container with Julia

[Singularity](https://docs.sylabs.io/guides/3.5/user-guide/introduction.html) is a tool for running code in containers (similar to Docker). A **container** is a piece of software representing the "user space" of a computer. It contains everything needed to run a piece of software, including an operating system environment and all of the necessary dependencies.

To run Julia on FASRC, you will need to set up a Singularity container with Julia installed. To obtain a container with the latest version of julia, log in to FASRC from the command line and run

```
singularity pull docker://julia
```

This will download a Singularity container with the latest version of Julia installed. We can now customize the container to include any additional packages we need. 

## 2: Install necessary packages on a compute node

To install packages in the Singularity container, you will need to run the container on a compute node. We can start an interactive session on a compute node using the `salloc` command. For example, to start an interactive session on the `test` partition with 1GB of memory and a time limit of 6 hours, run

```
salloc -p test --mem 1G -t 0-06:00
```

Once an interactive session has begun, we can enter the Singularity container with our latest version of Julia by running

```
singularity shell julia_latest.sif`
```

One could run the command above on the login node, without entering an interactive session. However, because Julia packages often take a long time to compile or update, it is recommended to use `salloc` to give yourself more memory so that the package installation process is not killed by the login node system. 

Once inside, run `julia` to start up the Julia REPL. Then, install necessary packages needed to run your code using the standard Julia REPL commands (i.e. type `]` to enter the package manager and `add PackageName`).

Once you have installed all necessary packages, you can exit the Julia REPL by typing `exit()`, and exit the Singularity container by typing `exit`.

## 3: Run a job

Now that you have set up a Singularity container with Julia and installed all necessary packages, you can write your Julia code in a `.jl` file and submit it as a job to the cluster. Simply add the line `singularity exec julia_latest.sif julia 'path/to/your/script.jl'` to your SLURM script; this will run the specified script using the Julia command. Other notes:

- One can modify the `julia` command using the [standard Julia command line arguments](https://docs.julialang.org/en/v1/manual/command-line-interface/).
- Use `srun -c $SLURM_CPUS_PER_TASK` and `julia --threads 8` if running a multi-threaded job (change "8" to however many threads you intend to use, but note that asking for more will use up more fairshare score and increase the time it takes for your job to be accepted). 

A full SLURM script that runs Julia with multiple threads might look like this; an actual script is included in the same directory as this document:

```
#!/bin/bash
#SBATCH -J sim_name
#SBATCH -o sim_name%j.out
#SBATCH -e sim_name%j.err
#SBATCH -p hejazi
#SBATCH -t 0-6:00:00
#SBATCH -c 8
#SBATCH --mem-per-cpu=10000

# Put what you want to do with singularity below
srun -c $SLURM_CPUS_PER_TASK singularity exec julia_latest.sif julia --threads 8 'path/to/script.jl' 'arg1' 'arg2'
```

Finally, once you've written a SLURM script "scriptname.sh", submit it as a job by running `sbatch scriptname.sh`. 
