# pkill -fe loky
# /media/data/conda/dlinsley/envs/powerAIlab/bin/python parallel_cube_merged_wkv.py

# Delete server data
# touch /cifs/data/tserre/CLPS_Serre_Lab/connectomics/cubed_mag1/delete_me

# Delete the lrs cubes
rm -rf /cifs/data/tserre/CLPS_Serre_Lab/connectomics/cubed_mag1/merge_data_wkw/*

# Delete current cubes
rm -rf /gpfs/data/tserre/data/wkcube/merge_data_wkw 

# Delete current skeletonization
rm -rf /gpfs/data/tserre/data/wkcube/skeleton_merge

# Delete current compresson
rm -rf /gpfs/data/tserre/data/wkcube_compress

# Run the cubing
/media/data/conda/dlinsley/envs/wkcuber/bin/python parallel_cube_merged_wkv.py
python parallel_cube_merged_wkv.py

# Skeletonize
export PYTHONPATH=$PYTHONPATH:/users/dlinsley/wkcuber/
rm -rf /gpfs/data/tserre/data/wkcube/skeleton_merge
cp /cifs/data/tserre/CLPS_Serre_Lab/connectomics/cubed_mag1/pbtest/ding/datasource-properties.json /gpfs/data/tserre/data/wkcube/
/media/data/anaconda3-ibm/bin/python merge_skeltons_in_wkw.py --input=/gpfs/data/tserre/data/wkcube --layer_name=merge_data_wkw --nml=/users/dlinsley/ffn_membrane/All_Skels_to_Stitch_Segs.nml  --output=/gpfs/data/tserre/data/wkcube/skeleton_merge
# /media/data/anaconda3-ibm/bin/python merge_skeltons_in_wkw.py --input=/gpfs/data/tserre/data/wkcube --layer_name=merge_data_wkw --nml=/users/dlinsley/ffn_membrane/skeletons.nml --output=/gpfs/data/tserre/data/wkcube/skeleton_merge

# Compress wkws
export PYTHONPATH=$PYTHONPATH:/users/dlinsley/wkcuber/
mkdir /gpfs/data/tserre/data/wkcube_compress
python -m wkcuber.compress --layer merge_data_wkw /gpfs/data/tserre/data/wkcube/skeleton_merge /gpfs/data/tserre/data/wkcube_compress
# python -m wkcuber.compress --layer merge_data_wkw /gpfs/data/tserre/data/wkcube /gpfs/data/tserre/data/wkcube_compress

# Mv the compressed cubes and sync
rsync -r --progress /gpfs/data/tserre/data/wkcube_compress/merge_data_wkw/1 /cifs/data/tserre/CLPS_Serre_Lab/connectomics/cubed_mag1/merge_data_wkw/
touch /cifs/data/tserre/CLPS_Serre_Lab/connectomics/cubed_mag1/sync_me

