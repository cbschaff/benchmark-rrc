#!/usr/bin/env bash

image=$1
logdir=$2
mkdir -p ${logdir}/out/user

echo "copying files..."
cp ${logdir}/user/custom_data.db ${logdir}/out/user/
cp ${logdir}/*.dat ${logdir}/out/user/

pushd .
script_dir=$(cd `dirname $0` && pwd)
work_dir=${script_dir}/../..
popd

${work_dir}/run_to_replay.py --backend-image ${image} --repository ${work_dir} --output-dir ${logdir}/out
mv ${logdir}/out/user/comparison.avi ${logdir}/comparison.avi
# ffmpeg -i ${logdir}/out/user/comparison.avi ${logdir}/comparison.mp4
rm -r ${logdir}/out
