#!/bin/bash
image=$1
logdir=$2
dir=`dirname $0`
${image} ros2 run robot_fingers robot_log_dat2csv ${logdir}/robot_data.dat ${logdir}/robot_data.csv
${image} python3 ${dir}/plot_scripts/finger_position.py ${logdir}/robot_data.csv
${image} python3 ${dir}/plot_scripts/plot.py ${logdir}/robot_data.csv ${logdir}/plot_action_repeat.pdf status_action_repetitions
