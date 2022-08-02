#!/bin/bash
#SBATCH --partition=general                   # Name of partition
#SBATCH --ntasks=16                           # Request 48 CPU cores
#SBATCH --exclude=cn[66-69,71-136,153-256,265-320,325-328]
#SBATCH --time=02:00:00                       # Job should run for up to 2 hours (for example)
#SBATCH --mail-type=END                       # Event(s) that triggers email notification (BEGIN,END,FAIL,ALL)
#SBATCH --mail-user=first.last@uconn.edu      # Destination email address

# Purge modules and install the correct versions of tensorflow

module purge
module load gcc/9.2.0 libffi/3.2.1 bzip2/1.0.6 tcl/8.6.6.8606 sqlite/3.30.1 lzma/4.32.7 
module load python/3.9.2

python3 --version

pip install --user --upgrade pip
pip install --user scipy
pip show --user scipy
pip install --user csv

python3 FGF_Classes/SciPy_QMC_test.py
