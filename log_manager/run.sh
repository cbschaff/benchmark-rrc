#!/bin/bash
if (( $# != 2 ))
then
    echo "Invalid number of arguments."
    echo "Usage:  $0 <log directory> <singularity image>"
    exit 1
fi
logdir=$1
image=$2
dir=`dirname $0`

hostname=robots.real-robot-challenge.com
username=`cat ${dir}/user.txt | head -n 1`

jobs=$(ssh -T -i ${dir}/sshkey ${username}@${hostname} <<< history | tail -n +2 | tr -s " " | awk -F'[. ]' '{print $1, $7}')

echo ${jobs} | xargs -n 2 echo
echo ${jobs} | xargs -t -n 2 -P 10 bash ${dir}/after_job.sh ${logdir} ${image}
