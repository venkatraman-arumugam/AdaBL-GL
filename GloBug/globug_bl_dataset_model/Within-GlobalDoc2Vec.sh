#!/bin/bash
#SBATCH --account=def-m2nagapp
#SBATCH --ntasks=1               # number of MPI processes
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=32G      # memory; default unit is megabytes
#SBATCH --time=18:00:00           # time (DD-HH:MM)
#SBATCH --output=%x-%j.out
#SBATCH --mail-user=w6s8l3k3q8t9b7d2@uw-swag.slack.com
#SBATCH --mail-type=ALL
source ~/replication/bin/activate
python Within-GlobalDoc2Vec.py