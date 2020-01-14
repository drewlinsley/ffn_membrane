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

