# Run 3x to test chaining
python db_tools.py --reset_config --reset_priority --priority_list=db/test_priorities.csv
CUDA_VISIBLE_DEVICES=6 python run_job.py
CUDA_VISIBLE_DEVICES=6 python run_job.py
CUDA_VISIBLE_DEVICES=6 python run_job.py
# CUDA_VISIBLE_DEVICES=6 python run_job.py
# CUDA_VISIBLE_DEVICES=6 python run_job.py
# CUDA_VISIBLE_DEVICES=6 python run_job.py

