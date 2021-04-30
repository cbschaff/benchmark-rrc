#!/bin/bash

. ./funcs.sh
wait_for_job_submission
fetch_and_upload_roboch_file
submit_job
wait_for_job_start ${job_id}
wait_for_job_finish ${job_id}
restore_roboch_file
