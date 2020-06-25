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

# Prep the DB for membrane detection
python db_tools.py --init_db --populate_db --segmentation_grid=9,9,3

# Prepare synapse data for training/testing
python synapse_saver.py

# Test a membrane detection
CUDA_VISIBLE_DEVICES=5 python hybrid_inference.py --membrane_only --path_extent=3,9,9 --membrane_slice=64,384,384

# Restore a DB from a dump
psql -h 127.0.0.1 -d connectomics connectomics < db_dumps/2_21_20.dump

# Restore synapse DB
python db_tools.py --populate_synapses --reset_synapses

# Get synapse detections
python synapse_test.py --keep_processing --pull_from_db

# Get synapse detections on a specific volume
python synapse_test.py --segmentation_path=25,30,43

# Convert synapse detections to webknossos
python pull_and_convert_predicted_synapses.py

# Fix synapse nml file
python synapses/fix_unlinked.py synapses/ribbon_ding.nml synapses/ribbon_ding_fixed.nml
python synapses/fix_unlinked.py synapses/amacrine_ding.nml synapses/amacrine_ding_fixed.nml

# Convert npy files to wk
sudo vi /usr/local/lib/python3.7/dist-packages/wkcuber/convert_nifti.py

python3.7 -m wkcuber.convert_nifti --color_file one_nifti --segmentation_file another_nifti --scale 13.2,13.2,26.0 /media/data_cifs/connectomics/mag1_membranes_nii/ /media/data_cifs/connectomics/cubed_membranes/

python3.7 -m wkcuber.convert_npy --scale 13.2,13.2,26.0 --source_path /media/data_cifs/connectomics/merge_data/ --target_path /media/data_cifs/connectomics/cubed_merge/  --path_storage /media/data_cifs/connectomics/merge_paths.npy

# Convert knossos to webknossos (if using a segmentation, make sure to copy over the original knossos meta files)
python3.7 -m wkcuber.convert_knossos --mag 1 --layer_name segmentation --dtype uint32 /media/data_cifs/connectomics/merge_data_nii_raw_v2 /media/data_cifs/connectomics/cubed_knossos_segmentations

