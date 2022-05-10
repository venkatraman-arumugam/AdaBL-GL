#!/bin/bash
#SBATCH --account=def-m2nagapp
#SBATCH --ntasks=1               # number of MPI processes
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=64G      # memory; default unit is megabytes
#SBATCH --time=5:00:00         # time (DD-HH:MM)
#SBATCH --output=%x-%j.out
#SBATCH --mail-user=XXXXXX
#SBATCH --mail-type=ALL
source ~/replication/bin/activate
python bl_dataset_dfg_graph.py