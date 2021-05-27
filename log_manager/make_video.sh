#!/bin/bash
if (( $# != 2 ))
then
    echo "Invalid number of arguments."
    echo "Usage:  $0 <singularity image> <log directory>"
    exit 1
fi
image=$1
logdir=$2
dir=`dirname $0`
${image} ros2 run trifinger_object_tracking tricamera_log_converter ${logdir}/camera_data.dat ${logdir}/video60.avi -c camera60
${image} ros2 run trifinger_object_tracking tricamera_log_converter ${logdir}/camera_data.dat ${logdir}/video180.avi -c camera180
${image} ros2 run trifinger_object_tracking tricamera_log_converter ${logdir}/camera_data.dat ${logdir}/video300.avi -c camera300

ffmpeg -i ${logdir}/video60.avi -i ${logdir}/video180.avi -filter_complex hstack -q:v 1 ${logdir}/video_temp.avi
ffmpeg -i ${logdir}/video_temp.avi -i ${logdir}/video300.avi -filter_complex hstack -q:v 1 ${logdir}/video.avi
ffmpeg -i ${logdir}/video.avi ${logdir}/video.webm

rm ${logdir}/video60.avi
rm ${logdir}/video180.avi
rm ${logdir}/video300.avi
rm ${logdir}/video_temp.avi
rm ${logdir}/video.avi
singularity run --nv ${image} python3 ${dir}/replay_scripts/replay.py ${logdir} ${logdir}/replay.avi
ffmpeg -i ${logdir}/replay.avi ${logdir}/replay.webm
rm ${logdir}/replay.avi
