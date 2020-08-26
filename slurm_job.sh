#!/bin/bash

#SBATCH --time=60:00:00
#SBATCH -p gpu --gres=gpu:1
#SBATCH -n 4
#SBATCH --mem=94G
#SBATCH --account=carney-tserre-condo
#SBATCH -C quadrortx
#SBATCH -J connectomics_reconstruction

# Specify an output file
# #SBATCH -o $0
# #SBATCH -e $1

# # Set up the environment by loading modules
# module load anaconda/3-5.2.0
. /gpfs/runtime/opt/anaconda/3-5.2.0/etc/profile.d/conda.sh
conda activate /users/dlinsley/anaconda/connectomics
python run_job.py --merge_segment_only --data_path=/cifs/data/tserre/CLPS_Serre_Lab/connectomics
echo FINISHED

