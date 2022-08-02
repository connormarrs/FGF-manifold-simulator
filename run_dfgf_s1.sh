#!/bin/bash
#SBATCH --partition=general                   # Name of partition
#SBATCH --ntasks=4                           # Request 48 CPU cores
#SBATCH --exclude=cn[66-69,71-136,153-256,265-320,325-328]
#SBATCH --time=02:00:00                       # Job should run for up to 2 hours (for example)
#SBATCH --mail-type=END                       # Event(s) that triggers email notification (BEGIN,END,FAIL,ALL)
#SBATCH --mail-user=first.last@uconn.edu      # Destination email address

# Purge modules and install the correct versions of tensorflow

module purge
module load cuda/8.0.61 cudnn/6.0 sqlite/3.18.0 tcl/8.6.6.8606 python/3.6.1

python --version

pip install --user --upgrade pip
python -m pip show pandas

python FGF_Classes/DFGF_S1.py
