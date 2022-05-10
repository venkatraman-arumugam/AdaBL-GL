#!/bin/bash
#SBATCH --account=def-m2nagapp
#SBATCH --ntasks=1               # number of MPI processes
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=64G      # memory; default unit is megabytes
#SBATCH --time=18:20:00           # time (DD-HH:MM)
#SBATCH --output=%x-%j.out
#SBATCH --mail-user=XXXXXX
#SBATCH --mail-type=ALL
source ~/replication/bin/activate
python main.py --data_setting cross_project_and_within_project -p