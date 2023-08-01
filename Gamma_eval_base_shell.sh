#!/bin/bash
#SBATCH -c 1
#SBATCH -t 1-12
#SBATCH --mem=1500                 
#SBATCH -p medium
#SBATCH -e hostname_%j.err   
#SBATCH --mail-type=FAIL 
#SBATCH --mail-user=octavioustalbot@g.harvard.edu

module load gcc/9.2.0 python/3.9.14

source /home/ot25/mypy/bin/activate

# $1 here will be var1
python Gamma_eval.py $1 $2 $3 $4 $5 $6