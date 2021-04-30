#!/bin/bash

# This is an example script how you can send submissions to the robot in a
# somewhat automated way.  The basic idea is the following:
#
#  1. submit job to robot
#  2. wait until job is finished
#  3. download relevant data from the run
#  4. run some algorithm on the data (e.g. to compute a new policy)
#  5. update the policy parameters in the git repository and push the
#     changes
#  6. goto 1 unless some termination criterion is fulfilled
#
# You can use the script mostly as is, you would just need to add your code for
# processing the data and pushing new parameters to git below.  You can also
# adapt any part of the script to your needs.  However, to avoid overloading
# our system, you MUST NOT poll the server at a higher rate than once per
# minute to see the status of the job!

# prompt for username and password (to avoid having user credentials in the
# bash history)
username=$2
password=$3
# there is no automatic new line after the password prompt
echo


# URL to the webserver at which the recorded data can be downloaded
base_url=https://robots.real-robot-challenge.com/output/${username}/data


# Check if the file/path given as argument exists in the "data" directory of
# the user
function curl_check_if_exists()
{
    local filename=$1

    http_code=$(curl -sI --user ${username}:${password} -o /dev/null -w "%{http_code}" ${base_url}/${filename})

    case "$http_code" in
        200|301) return 0;;
        *) return 1;;
    esac
}


# Send 1 single submission
for (( i=0; i<1; i++))
do
    echo "Submit job"
    submit_result=$(sshpass -p ${password} ssh -T ${username}@robots.real-robot-challenge.com <<< submit)
    job_id=$(echo ${submit_result} | grep -oP 'job\(s\) submitted to cluster \K[0-9]+')
    if [ $? -ne 0 ]
    then
        echo "Failed to submit job.  Output:"
        echo "${submit_result}"
        sleep 100
    else

      echo "Submitted job with ID ${job_id}"

      echo "Wait for job to be started"
      job_started=0
      # wait for the job to start (abort if it did not start after 10 min)
      for (( i=0; i<30; i++))
      do
          # Do not poll more often than once per minute!
          sleep 20

          # wait for the job-specific output directory
          if curl_check_if_exists ${job_id}
          then
              job_started=1
              break
          fi
          date
      done

      if (( ${job_started} == 0 ))
      then
          echo "Job did not start."
          exit 1
      else

        echo "Job is running.  Wait until it is finished"
        # if the job did not finish 10 minutes after it started, something is
        # wrong, abort in this case
        job_finished=0
        for (( i=0; i<30; i++))
        do
            # Do not poll more often than once per minute!
            sleep 20

            # report.json is explicitly generated last of all files, so once this
            # exists, the job has finished
            if curl_check_if_exists ${job_id}/report.json
            then
                job_finished=1
                break
            fi
            date
        done

        if (( ${job_finished} == 0 ))
        then
            echo "Job did not finish in time."
            sleep 10
        fi

        echo "Job ${job_id} finished."

        job_dir=$1
        echo "Download data to ${job_dir}"

        sleep 10
        # Download data.  Here only the report file is downloaded as example.  Add
        # equivalent commands for other files as needed.
        curl --user ${username}:${password} -o "${job_dir}/report.json" ${base_url}/${job_id}/report.json
        curl --user ${username}:${password} -o "${job_dir}/user_stdout.txt" ${base_url}/${job_id}/user_stdout.txt
        curl --user ${username}:${password} -o "${job_dir}/reward.pkl" ${base_url}/${job_id}/user/reward.pkl
        curl --user ${username}:${password} -o "${job_dir}/goal.json" ${base_url}/${job_id}/user/goal.json
        curl --user ${username}:${password} -o "${job_dir}/stdout.txt" ${base_url}/../stdout.txt
        curl --user ${username}:${password} -o "${job_dir}/stderr.txt" ${base_url}/../stderr.txt
        curl --user ${username}:${password} -o "${job_dir}/robot_data.dat" ${base_url}/${job_id}/robot_data.dat
        curl --user ${username}:${password} -o "${job_dir}/camera_data.dat" ${base_url}/${job_id}/camera_data.dat
        echo ${job_id} >> "${job_dir}/job_id.txt"

        # if there was a problem with the backend, download its output and exit
        if grep -q "true" "${job_dir}/report.json"
        then
            echo "ERROR: Backend failed!  Download backend output and stop"
            sleep 10
        fi
        sleep 10


        fi
    fi

done
