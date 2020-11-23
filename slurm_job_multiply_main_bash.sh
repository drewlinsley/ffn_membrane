#!/bin/bash

echo How many jobs do you want to launch?
read num_jobs

echo Starting index  # How many jobs do you want to launch?
read start_index

up_num_jobs=$((num_jobs + start_index))

for (( i=start_index; i<$up_num_jobs; i++ ))
do
    sbatch --error=slurm_outputs/err_{$i}.out --output=slurm_outputs/out_{$i}.out slurm_job_main.sh
done

