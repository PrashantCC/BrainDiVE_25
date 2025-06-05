#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --mem=100G
#SBATCH --cpus-per-task=4
#SBATCH --open-mode=append  
#SBATCH --output=./sbatch_output/output-%A-%x-%u.out 
#SBATCH --time=8-00:00:00


# source ~/myenv/bin/activate

# Set the ROOT environment variable to your home directory
ROOT=~/BrainDiVE/

# Put the code directory on your python path
PYTHONPATH=:${ROOT}modfit/code/:${PYTHONPATH}
echo $PYTHONPATH

# Go to the folder where the script is located
cd ${ROOT}modfit/code/utils/run/

# To test the code, use debug=1
# To run for real, set debug=0 (False)
debug=1

python3 prep_data.py --debug $debug


