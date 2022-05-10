#!/bin/bash
#SBATCH --account=def-m2nagapp
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=64G      # memory; default unit is megabytes
#SBATCH --time=04:00:00           # time (DD-HH:MM)
#SBATCH --mail-user=w6s8l3k3q8t9b7d2@uw-swag.slack.com
#SBATCH --mail-type=ALL
module load java
source ~/replication/bin/activate
python sc_glob_pysparl_prcessing.py