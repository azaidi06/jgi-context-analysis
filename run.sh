#!/bin/bash

#SBATCH --account=m342_g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=1
#SBATCH --qos=regular
#SBATCH --time=02:00:00
#SBATCH --constraint=gpu&hbm80


module load python
conda activate dl24

srun python run.py
