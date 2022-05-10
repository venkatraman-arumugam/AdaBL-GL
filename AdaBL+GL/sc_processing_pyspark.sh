#!/bin/bash
#SBATCH --account=def-m2nagapp
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=32G      # memory; default unit is megabytes
#SBATCH --time=05:00:00           # time (DD-HH:MM)
#SBATCH --output=%x-%j.out
#SBATCH --mail-user=XXXXXX
#SBATCH --mail-type=ALL
module load java
source ~/replication/bin/activate
python sc_processing_pyspark.py