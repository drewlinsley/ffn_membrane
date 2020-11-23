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
# /media/data/conda/dlinsley/envs/powerAIlab/bin/python v2_bu_h_perform_merge_nii.py
bash run_merge.sh

# Convert segmentations to wK cubes
# python parallel_cube_merged_wkv.py
# mv /cifs/data/tserre_lrs/projects/prj_connectomics/connectomics_data/merge_data_wkw/1 /cifs/data/tserre_lrs/connectomics/cubed_mag1/merge_data_wkw
bash run_cubing.sh
rm -rf /cifs/data/tserre/CLPS_Serre_Lab/connectomics/cubed_mag1/merge_data_wkw/*

# Compress wkws
rm -rf /gpfs/data/tserre/data/wkcube_compress
python -m wkcuber.compress --layer merge_data_wkw /gpfs/data/tserre/data/wkcube /gpfs/data/tserre/data/wkcube_compress

# Move compressed wkws to LRS
# mkdir /cifs/data/tserre/CLPS_Serre_Lab/connectomics/cubed_mag1/merge_data_wkw/1
# rsync -avzh --progress /gpfs/data/tserre/data/wkcube_compress/merge_data_wkw/1 /cifs/data/tserre/CLPS_Serre_Lab/connectomics/cubed_mag1/merge_data_wkw/
rsync -r --progress /gpfs/data/tserre/data/wkcube_compress/merge_data_wkw/1/ /cifs/data/tserre/CLPS_Serre_Lab/connectomics/cubed_mag1/merge_data_wkw
touch /cifs/data/tserre/CLPS_Serre_Lab/connectomics/cubed_mag1/sync_me

# Merge skeletons in the wkw -- DEPRECIATED
export PYTHONPATH=$PYTHONPATH:/users/dlinsley/wkcuber/
# cp /cifs/data/tserre/CLPS_Serre_Lab/connectomics/cubed_mag1/pbtest/ding/datasource-properties.json /gpfs/data/tserre/data/wkcube
# /media/data/anaconda3-ibm/bin/python merge_skeltons_in_wkw.py --input=/gpfs/data/tserre/data/wkcube --layer_name=merge_data_wkw --nml=/users/dlinsley/ffn_membrane/skeletons.nml  --output=/gpfs/data/tserre/data/wkcube/skeleton_merge
cp /cifs/data/tserre/CLPS_Serre_Lab/connectomics/cubed_mag1/pbtest/ding/datasource-properties.json /gpfs/data/tserre/data/wkcube_compress/
/media/data/anaconda3-ibm/bin/python merge_skeltons_in_wkw.py --input=/gpfs/data/tserre/data/wkcube_compress --layer_name=merge_data_wkw --nml=/users/dlinsley/ffn_membrane/skeletons.nml  --output=/gpfs/data/tserre/data/wkcube/skeleton_merge

#### N.B.
Files are written to /users/dlinsley/scratch/connectomics_data/. A cron job rsyncs daily to /cifs/data/tserre_lrs/projects/prj_connectomics/connectomics_data_scratch
crontab -e
0 18 * * * rsync --ignore-existing-files --progress -a /users/dlinsley/scratch/connectomics_data/mag1_merge_segs /cifs/data/tserre_lrs/projects/prj_connectomics/connectomics_data_scratch/mag1_merge_segs
0 20 * * * rsync --ignore-existing-files --progress -a /users/dlinsley/scratch/connectomics_data/ding_segmentations_merge /cifs/data/tserre_lrs/projects/prj_connectomics/connectomics_data_scratch/ding_segmentations_merge
0 22 * * * rsync --ignore-existing-files --progress -a /users/dlinsley/scratch/connectomics_data/mag1_membranes /cifs/data/tserre_lrs/projects/prj_connectomics/connectomics_data_scratch/mag1_membranes

Total directories that need to be checked for files:
- /users/dlinsley/scratch/connectomics_data
- /cifs/data/tserre_lrs/projects/prj_connectomics/connectomics_data
- /cifs/data/tserre_lrs/projects/prj_connectomics/connectomics_data_v0
- /cifs/data/tserre_lrs/projects/prj_connectomics/connectomics_data_scratch
- /cifs/data/tserre_lrs/connectomics

