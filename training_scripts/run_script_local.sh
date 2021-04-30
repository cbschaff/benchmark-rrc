#! /bin/bash
image=$1
config=$2
rrc_root="$( cd "$( dirname "$( dirname "${BASH_SOURCE[0]}" )" )" &> /dev/null && pwd )"
echo ${rrc_root}
echo ${image}
echo ${config}

cd $rrc_root
echo `date`
mounts=`python3 training_scripts/build_ws.py ${rrc_root} ${image} ${config}`
echo ${mounts}
echo `date`
singularity exec --contain --nv -B ${mounts} ${image} bash -c \
        ". /setup.bash; . /ws/devel/setup.bash; timeout -s SIGINT -k 0.1h --foreground 3.5h python3 /ws/src/usercode/python/residual_learning/train.py /logdir/singularity_config.yaml"
