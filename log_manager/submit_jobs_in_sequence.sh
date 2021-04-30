#!/usr/bin/env bash

num_jobs=$1
for (( i=0; i<${num_jobs}; i++))
do
    echo "======== Processing job [${i}/${num_jobs}] ========"
    ./submit_job.sh
done
