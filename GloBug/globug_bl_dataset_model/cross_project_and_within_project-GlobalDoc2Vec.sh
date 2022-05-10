#!/bin/bash
#SBATCH --account=def-m2nagapp
#SBATCH --ntasks=1               # number of MPI processes
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=64G      # memory; default unit is megabytes
#SBATCH --time=18:20:00           # time (DD-HH:MM)
#SBATCH --output=%x-%j.out
#SBATCH --mail-user=w6s8l3k3q8t9b7d2@uw-swag.slack.com
#SBATCH --mail-type=ALL
source ~/replication/bin/activate
python cross_project_and_within_project-GlobalDoc2Vec.py