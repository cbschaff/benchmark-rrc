#!/bin/bash
if (( $# != 4 ))
then
    echo "Invalid number of arguments."
    echo "Usage:  $0 <log directory> <singularity image> <job id> <job status>"
    exit 1
fi
logdir=$1
image=$2
jobid=$3
status=$4
dir=`dirname $0`

if [ -d ${logdir}/${jobid} ]; then
    exit
fi

if [ ! ${status} == 'C' ]; then
    exit
fi

bash ${dir}/download_logs.sh ${jobid} ${logdir}
singularity run --nv ${image} python3 ${dir}/replay_scripts/compute_reward.py ${logdir}/${jobid}
bash ${dir}/make_plots.sh ${image} ${logdir}/${jobid}
bash ${dir}/make_video.sh ${image} ${logdir}/${jobid}
