#!/usr/bin/env bash
# prompt for username and password (to avoid having user credentials in the
# bash history)
username=`cat user.txt | head -n 1`
password=`cat user.txt | sed -n '2 p'`

# URL to the webserver at which the recorded data can be downloaded
hostname=robots.real-robot-challenge.com
data_url=https://${hostname}/output/${username}/data

# Check if the file/path given as argument exists in the "data" directory of
# the user
function curl_check_if_exists()
{
    local filename=$1

    http_code=$(curl -sI --user ${username}:${password} -o /dev/null -w "%{http_code}" ${data_url}/${filename})

    case "$http_code" in
        200|301) return 0;;
        *) return 1;;
    esac
}

function check_if_job_submittable()
{
    status=$(ssh -T -i sshkey ${username}@${hostname} <<<status)
    empty_when_no_jobs=$(echo ${status} | sed 's/[^1-9]*//g')
    # echo ${status}
    # echo "empty when no jobs: ${empty_when_no_jobs}"
    if [ -z "${empty_when_no_jobs}" ]
    then
        # echo "jobs are submittable"
        return 0
    else
        # echo "jobs are NOT submittable"
        return 1
    fi
}

function check_if_backend_exited_normally()
{
    # URL to the webserver at which the recorded data can be downloaded
    base_url=https://robots.real-robot-challenge.com/output/${username}/data
    file='report.json'

    curl --silent --user ${username}:${password} ${base_url}/${job_id}/${file} > /tmp/${file}
    if grep -q "true" "/tmp/${file}"
    then
        echo "WARNING: Backend failed!"
        return 1
    else
        return 0
    fi
}

function submit_job()
{
    echo "Submit job"
    submit_result=$(ssh -T -i sshkey ${username}@${hostname} <<<submit)
    job_id=$(echo "${submit_result}" | cut -d' ' -f 6 | grep -oE '[0-9]+')
    if [ $? -ne 0 ]
    then
        echo "Failed to submit job.  Output:"
        echo "${submit_result}"
        echo "Job ID ${job_id}"
        exit 1
    fi
    echo "Submitted job with ID ${job_id}"

}

function wait_for_job_submission()
{
    for (( i=0; i<60; i++))
    do
        sleep 10
        if check_if_job_submittable
        then
            echo "check_if_job_submittable returned True"
            break
        else
            echo "check_if_job_submittable returned False"
            echo "The server is not ready yet... (retrying in 10 seconds)"
        fi
    done
}

function wait_for_job_start()
{
    echo "Waiting for the job ${job_id} to be started"
    local job_started=0
    # wait for the job to start (abort if it did not start after half an hour)
    for (( i=0; i<30; i++))
    do
        # Do not poll more often than once per minute!
        sleep 60

        # wait for the job-specific output directory
        if curl_check_if_exists $1
        then
            local job_started=1
            break
        fi
        date
    done
    if (( ${job_started} == 0 ))
    then
        echo "Job did not start."
        exit 1
    fi

    echo "Job is running.  Wait until it is finished"
}

function wait_for_job_finish()
{
    # if the job did not finish 10 minutes after it started, something is
    # wrong, abort in this case
    local job_finished=0
    for (( i=0; i<15; i++))
    do
        # Do not poll more often than once per minute!
        sleep 60

        # report.json is explicitly generated last of all files, so once this
        # exists, the job has finished
        if curl_check_if_exists $1/report.json && check_if_job_submittable
        then
            local job_finished=1
            echo "The job ${job_id} has finished."
            check_if_backend_exited_normally ${job_id}
            break
        fi
        date
    done
}

function fetch_and_upload_roboch_file()
{
    scp -i sshkey ${username}@${hostname}:roboch.json roboch.json.bk
    scp -i sshkey roboch.json ${username}@${hostname}:roboch.json
}

function restore_roboch_file()
{
    scp -i sshkey roboch.json.bk ${username}@${hostname}:roboch.json
}
