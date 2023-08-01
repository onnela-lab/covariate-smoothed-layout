#!/bin/bash
#SBATCH -c 1
#SBATCH -t 0-4
#SBATCH --mem=1000                 
#SBATCH -p short
#SBATCH -e hostname_%j.err   
#SBATCH --mail-type=FAIL 
#SBATCH --mail-user=octavioustalbot@g.harvard.edu

module load gcc/9.2.0 python/3.9.14

source /home/ot25/mypy/bin/activate

# $1 here will be var1
python Plot_positions.py