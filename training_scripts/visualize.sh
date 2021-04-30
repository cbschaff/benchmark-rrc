#! /bin/bash
image=$1
# remove trailing slash
logdir=${2%/}
rrc_root="$( cd "$( dirname "$( dirname "${BASH_SOURCE[0]}" )" )" &> /dev/null && pwd )"
cd ${rrc_root}
echo `date`

mounts="${logdir}/catkin_ws:/ws,${logdir}/logs:/logdir"
echo ${mounts}
echo `date`
script_path=src/usercode/python/residual_learning/viz.py

singularity exec --contain --nv -B ${mounts} ${image} bash -c \
        ". /setup.bash; . /ws/devel/setup.bash; python3 /ws/${script_path} ${*:3}"
