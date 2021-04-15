# pkill -fe loky
# /media/data/conda/dlinsley/envs/powerAIlab/bin/python parallel_cube_merged_wkv.py

# Delete server data
# touch /cifs/data/tserre/CLPS_Serre_Lab/connectomics/cubed_mag1/delete_me

# Delete the lrs cubes
rm -rf /cifs/data/tserre/CLPS_Serre_Lab/connectomics/cubed_mag1/merge_data_wkw/*

# Delete current skeletonization
rm -rf /gpfs/data/tserre/data/wkcube/skeleton_merge

# Skeletonize
export PYTHONPATH=$PYTHONPATH:/users/dlinsley/wkcuber/
rm -rf /gpfs/data/tserre/data/wkcube/skeleton_merge
cp /cifs/data/tserre/CLPS_Serre_Lab/connectomics/cubed_mag1/pbtest/ding/datasource-properties.json /gpfs/data/tserre/data/wkcube/
/media/data/anaconda3-ibm/bin/python merge_skeltons_in_wkw.py --input=/gpfs/data/tserre/data/wkcube --layer_name=merge_data_wkw --nml=/users/dlinsley/ffn_membrane/All_Skels_to_Stitch_Segs.nml  --output=/gpfs/data/tserre/data/wkcube/skeleton_merge
# /media/data/anaconda3-ibm/bin/python merge_skeltons_in_wkw.py --input=/gpfs/data/tserre/data/wkcube --layer_name=merge_data_wkw --nml=/users/dlinsley/ffn_membrane/skeletons.nml --output=/gpfs/data/tserre/data/wkcube/skeleton_merge

# # Create downsampled versions
# cp /cifs/data/tserre/CLPS_Serre_Lab/connectomics/cubed_mag1/pbtest/ding/datasource-properties.json /gpfs/data/tserre/data/wkcube/skeleton_merge/
# python -m wkcuber.downsampling /gpfs/data/tserre/data/wkcube/skeleton_merge/ --from_mag 1 --max 16 --layer_name merge_data_wkw

# # Mv downsampled wkws
# mv /gpfs/data/tserre/data/wkcube/skeleton_merge/merge_data_wkw/16-16-8 /gpfs/data/tserre/data/wkcube/skeleton_merge/merge_data_wkw/16
# mv /gpfs/data/tserre/data/wkcube/skeleton_merge/merge_data_wkw/8-8-4 /gpfs/data/tserre/data/wkcube/skeleton_merge/merge_data_wkw/8
# mv /gpfs/data/tserre/data/wkcube/skeleton_merge/merge_data_wkw/4-4-2 /gpfs/data/tserre/data/wkcube/skeleton_merge/merge_data_wkw/4
# mv /gpfs/data/tserre/data/wkcube/skeleton_merge/merge_data_wkw/2-2-1 /gpfs/data/tserre/data/wkcube/skeleton_merge/merge_data_wkw/2

# Compress wkws
mkdir /gpfs/data/tserre/data/wkcube_compress
python -m wkcuber.compress --layer merge_data_wkw /gpfs/data/tserre/data/wkcube/skeleton_merge /gpfs/data/tserre/data/wkcube_compress

# Mv the compressed cubes
# rsync -rzva -O --progress /gpfs/data/tserre/data/wkcube_compress/merge_data_wkw/* /cifs/data/tserre/CLPS_Serre_Lab/connectomics/cubed_mag1/merge_data_wkw/
msrsync -P -p 32 --stats --rsync "-rvz --no-perms" /gpfs/data/tserre/data/wkcube_compress/merge_data_wkw/* /cifs/data/tserre/CLPS_Serre_Lab/connectomics/cubed_mag1/merge_data_wkw/

# And sync
touch /cifs/data/tserre/CLPS_Serre_Lab/connectomics/cubed_mag1/sync_me

