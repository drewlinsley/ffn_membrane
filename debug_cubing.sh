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
mkdir /gpfs/data/tserre/data/wkcube/merge_data_wkw
/media/data/conda/dlinsley/envs/wkcuber/bin/python parallel_cube_merged_wkv.py  # Gathers data
python parallel_cube_merged_wkv.py

# Skeletonize
export PYTHONPATH=$PYTHONPATH:/users/dlinsley/wkcuber/
rm -rf /gpfs/data/tserre/data/wkcube/skeleton_merge
cp /cifs/data/tserre/CLPS_Serre_Lab/connectomics/cubed_mag1/pbtest/ding/datasource-properties.json /gpfs/data/tserre/data/wkcube/

# msrsync -P -p 32 --stats --rsync "-rvz --no-perms" /gpfs/data/tserre/data/wkcube/merge_data_wkw /gpfs/data/tserre/data/wkcube/merge_data_wkw/skeleton_merge

# Compress wkws
mkdir /gpfs/data/tserre/data/wkcube_compress
python -m wkcuber.compress --layer merge_data_wkw /gpfs/data/tserre/data/wkcube /gpfs/data/tserre/data/wkcube_compress

# Mv the compressed cubes
# rsync -rzva -O --progress /gpfs/data/tserre/data/wkcube_compress/merge_data_wkw/* /cifs/data/tserre/CLPS_Serre_Lab/connectomics/cubed_mag1/merge_data_wkw/
msrsync -P -p 32 --stats --rsync "-rvz --no-perms" /gpfs/data/tserre/data/wkcube_compress/merge_data_wkw/* /cifs/data/tserre/CLPS_Serre_Lab/connectomics/cubed_mag1/merge_data_wkw/

# And sync
touch /cifs/data/tserre/CLPS_Serre_Lab/connectomics/cubed_mag1/sync_me

