#! /bin/bash

if (( $# < 2 ))
then
    echo "Invalid number of arguments."
    echo "Usage: $0 <singularity_image.sif> <cmd>"
    exit
fi

rrc_root=`pwd`
rrc_image=$1
expdir=${rrc_root}/build
export SINGULARITYENV_DISPLAY=$DISPLAY

# build dir
if [ -d ${expdir}/catkin_ws ]
then
    rm -r ${expdir}
fi

# build catkin workspace
mkdir -p ${expdir}/catkin_ws/src/usercode
mkdir -p ${expdir}/logs
cp -r ${rrc_root}/python ${expdir}/catkin_ws/src/usercode
cp -r ${rrc_root}/*.json ${expdir}/catkin_ws/src/usercode
cp -r ${rrc_root}/*.xml ${expdir}/catkin_ws/src/usercode
cp -r ${rrc_root}/setup.py ${expdir}/catkin_ws/src/usercode
cp -r ${rrc_root}/setup.cfg ${expdir}/catkin_ws/src/usercode
cp -r ${rrc_root}/scripts ${expdir}/catkin_ws/src/usercode
cp -r ${rrc_root}/resource ${expdir}/catkin_ws/src/usercode
singularity exec --cleanenv --contain -B ${expdir}/catkin_ws:/ws ${rrc_image} bash -c ". /setup.bash; cd /ws; colcon build"

# run command
singularity exec --cleanenv --contain --nv -B ${expdir}/catkin_ws:/ws,${expdir}/logs:/logdir,/run,/dev ${rrc_image} bash -c ". /setup.bash; . /ws/install/setup.bash; ${*:2}"
