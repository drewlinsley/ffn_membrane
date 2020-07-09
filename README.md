## Scripts for reconstructing connectomes.
# Includes routines for prepping annotations for membranes/segments/synapses and training models for each of these sources
The key innovations are (1) using RNNs (fGRU/gammanet) for membrane detection, which yields high-quality and general predictions across the entire volume. (2) Introducing these membranes into FFN-segmentation (which uses a 1-ts gammanet), and synapse detection (which uses a U-Net for speed).

DBs are used to track progress and organize jobs. The volume is processed in parallel, with redundancy, then a routine is used to merge each z-axis plane independently ("horizontal merge") before propogating labels upward from level to level ("bottom-up merge"). These scripts are in the CCV branch.

Finally, data is packaged up for webknossos for visualization and curation. The webserver lives at https://connectomics.clps.brown.edu/

Team members are:
Drew Linsley
Paulo Baptista
Junkyung Kim
Thomas Serre
David Berson
And the rest of the Berson lab

##

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
bash prepare_synapse_data.sh

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

python3.7 -m wkcuber.convert_knossos --layer_name color /media/data_cifs/connectomics/mag16 /media/data_cifs/connectomics/cubed_mag16
python3.7 -m wkcuber.convert_knossos --layer_name color /media/data_cifs/connectomics/mag8 /media/data_cifs/connectomics/cubed_mag8
python3.7 -m wkcuber.convert_knossos --layer_name color /media/data_cifs/connectomics/mag4 /media/data_cifs/connectomics/cubed_mag4
python3.7 -m wkcuber.convert_knossos --layer_name color /media/data_cifs/connectomics/mag2 /media/data_cifs/connectomics/cubed_mag2

# Location of nii outputs
`self.nii_path_str = os.path.join(self.write_project_directory, 'mag1_segs/x%s/y%s/z%s/110629_k0725_mag1_x%s_y%s_z%s.nii')`
`self.nii_merge_path_str = os.path.join(self.write_project_directory, 'mag1_merge_segs/x%s/y%s/z%s/110629_k0725_mag1_x%s_y%s_z%s.nii')`
Note that there is a transposition between segmentations and .raw Ding files. XYZ -> ZYX. This is in the wkv conversion scripts.

# Compress WKW data.
python3.7 -m wkcuber.compress --layer merge_data_wkw /media/data_cifs/connectomics/cubed_mag1/
