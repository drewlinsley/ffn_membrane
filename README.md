# Install postgres
1. sudo apt update
2. sudo apt install postgresql postgresql-contrib

# Install python packages
1. `sudo apt-get install libpq-dev python-dev`
2. `python setup.py install`
3. `pip install -r requirements.txt`

# Segment a volume
- First time? `CUDA_VISIBLE_DEVICES=4 python flexible_hybrid_new_model_test.py`
- Resegmenting? `CUDA_VISIBLE_DEVICES=4 python flexible_hybrid_new_model_test.py --idx=<integer greater than 0]` 

# Access the DB
- psql connectomics -h 127.0.0.1 -d connectomics

# Request interact
interact -t 48:00:00 -n 2 -m 32g -q gpu -g 1 -a carney-tserre-condo -f quadrortx
# conda activate connectomics
# module load anaconda/3-5.2.0
# module load tensorflow/1.13.1_gpu
conda activate /users/dlinsley/anaconda/connectomics

#####
# Merge all the segmentations
conda activate /users/dlinsley/anaconda/connectomics
/media/data/conda/dlinsley/envs/powerAIlab/bin/python v2_bu_h_perform_merge_nii.py

# Convert segmentations to wK cubes
python parallel_cube_merged_wkv.py
mv /cifs/data/tserre_lrs/projects/prj_connectomics/connectomics_data/merge_data_wkw/1 /cifs/data/tserre_lrs/connectomics/cubed_mag1/merge_data_wkw

# Merge skeletons in the wkw
export PYTHONPATH=$PYTHONPATH:/users/dlinsley/wkcuber/
python merge_skeltons_in_wkw.py --input=/cifs/data/tserre_lrs/connectomics/cubed_mag1 --layer_name=merge_data_wkw --nml=/users/dlinsley/ffn_membrane/skeletons.nml --output=/cifs/data/tserre_lrs/connectomics/cubed_mag1_merged

