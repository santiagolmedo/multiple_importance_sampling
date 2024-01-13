#!/bin/bash
#SBATCH --job-name=mis
#SBATCH --ntasks=1
#SBATCH --mem=20480
#SBATCH --time=4:00:00
#SBATCH --tmp=2G
#SBATCH --partition=normal
#SBATCH --qos=normal
#SBATCH --mail-type=ALL
#SBATCH --mail-user=santiolmedo99@gmail.com

source /etc/profile.d/modules.sh

python mis.py
