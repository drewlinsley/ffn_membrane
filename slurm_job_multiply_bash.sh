#!/bin/bash

echo How many jobs do you want to launch?
read num_jobs

for (( i=1; i<=$num_jobs; i++ ))
do
    sbatch slurm_job.sh
done

