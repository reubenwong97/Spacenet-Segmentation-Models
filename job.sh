#!/bin/bash
#SBATCH --job-name=spacenet_6
#SBATCH --output=spacenet_6.out
#SBATCH --error=spacenet_6.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --ntasks-per-node=8
#SBATCH --qos=normal
#SBATCH --partition=SCSEGPU_UG

module load anaconda
conda activate tf2.1
python $1

