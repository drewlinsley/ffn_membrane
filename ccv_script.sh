#!/bin/bash

#SBATCH --time=200:00:00
#SBATCH -n 6
#SBATCH -p gpu
#SBATCH --mem=80GB
#SBATCH --gres=gpu:1
#SBATCH --constraint=DGX
#SBATCH --nodelist=gpu1210
##SBATCH --qos=pri-dlinsley
#SBATCH -J j16
#SBATCH -o j16_%j.out
#SBATCH -e j16_%j_e.out

cd /users/dlinsley/ffn_membrane
module load cuda/9.0.176 cudnn/7.0
module load tensorflow/1.5.0_gpu
source ~/venv/bin/activate
pip install scipy
pip install opencv-python
pip install scikit-image
pip install tqdm
pip install sklearn
pip install -r requirements.txt
nvidia-smi
python run_job.py --experiment=LMD_ft --ckpt=/gpfs/data/tserre/data/lung/checkpoints/resnet_18_colorization_LMD_pretrain_2019_07_02_10_07_57_141437/model_91650.ckpt-91650 --model=fine_tune_resnet_18_colorization

