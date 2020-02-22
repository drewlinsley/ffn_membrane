#!/bin/bash
# GPU worker for running experiments in your database. Continues until the DB is empty.

if [ -z $1 ]
then
    read -p "Enter the ID of the gpu you want to use: "  gpu
else
    gpu=$1
fi
echo "Developing worker for gpu $gpu."

export PGPASSWORD=connectomics
RUN_LOOP=true
while [ $RUN_LOOP == "true" ]
    do
        CUDA_VISIBLE_DEVICES=$gpu python run_job.py --segment_only
    done

