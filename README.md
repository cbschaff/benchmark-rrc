Benchmarking Structured Policies and Policy Optimization for Real-World Dexterous Object Manipulation
=============================================================

This is an **updated version** of the code used for the publication
"Benchmarking Structured Policies and Policy Optimization for Real-World
Dexterous Object Manipulation".  It is kept up-to-date so that it can be run
with the current version of the TriFinger robot software.  Beware that the
behaviour of the robot may be a bit different compared to the original version
(especially in simulation as the robot model was changed there).

**[arXiv](https://arxiv.org/abs/2105.02087) | [project website](https://sites.google.com/view/benchmark-rrc)**

## Quickstart

### Running the code in simulation

**Make sure to download the Phase 2 image**

To run the code locally, first install [Singularity](https://sylabs.io/guides/3.5/user-guide/quick_start.html)
and download [singularity image for Phase 2](https://people.tuebingen.mpg.de/felixwidmaier/realrobotchallenge/robot_phase/singularity.html#singularity-download-image)
from the Real Robot Challenge. A couple extra dependencies are required to run our code. To create the required singularity image, run:
```singularity build --fakeroot image.sif image.def```

Use the `run_locally.sh` script to build the catkin workspace and run commands
inside the singularity image.
For example, to run the Motion Planning with Planned Grasp (MP-PG) on a random goal of difficulty 4, use the following
command:
```bash
./run_locally.sh /path/to/singularity/image.sif ros2 run rrc run_local_episode.py 4 mp-pg
```

Use ```scripts/run_local_episode.py``` to visualize all of our implemented approaches. You can run our code with and without residual models and BO optimized parameters. See ```scripts/run_local_episode.py``` for the arguments.


### Running the code on the robot cluster

Similarly to simulation, ```scripts/run_episode.py``` can be used to run our methods on the real platform. Edit ```run``` to set the correct arguments.

For detailed instructions on how to run this code on the robot cluster, see [this](https://people.tuebingen.mpg.de/felixwidmaier/realrobotchallenge/robot_phase/submission_system.html) page.

This repository contains code for automatically submitting jobs and analyzing logs in the [log_manager](https://github.com/ripl-ttic/rrc_phase_3/tree/cleanup/log_manager) directory.


## Optimizing hyperparameters using BO

The functionality to run BO for optimizing the hyperparameters is contained inside `python/cic/bayesian_opt`. 
In this location, also a README file is provided which details how the experiments can be run.


## Improving controllers with Residual Policy Learning

We use a residual version of Soft Actor Critic to train residual controllers using [this Deep RL library](https://github.com/cbschaff/dl).
To train a controller for the MP-PG controller use the following command:

```bash
./training_scripts/run_script_local.sh /path/to/singularity/image ./training_scripts/configs/mp_pg.yaml
```

Similar configs for the other controllers exist in the ```training_scripts/configs``` folder.

To record or evaluate the model call the ```visualize.sh``` and ```evaluate.sh``` scripts respectively:


```bash
./training_scripts/visualize.sh /path/to/singularity/image /path/to/log/directory -n num_episodes -t ckpt

./training_scripts/evaluate.sh /path/to/singularity/image /path/to/log/directory -n num_episodes -t ckpt
```

## Contributions

The code is the joint work of [Niklas Funk](https://github.com/nifunk), [Charles Schaff](https://github.com/cbschaff), [Rishabh Madan](https://github.com/madan96), and [Takuma Yoneda](https://github.com/takuma-ynd).

The repository is structured as a catkin package and builds on the
[example package](https://github.com/rr-learning/rrc_example_package) provided by the Real Robot Challenge,
and [this planning library](https://github.com/yijiangh/pybullet_planning).

The paper and this repository combines the code and algorithms from three teams that participated in the Real Robot Challenge:
- TTI-Chicago: [Motion Planning with Planned Grasps](http://arxiv.org/abs/2101.02842)
- University of Washington: [Cartesian Position Control with a Triangulated Grasp](https://openreview.net/pdf?id=9tYX-lukeq)
- TU Darmstadt: [Cartesian Impedance Control with Centered Grasps](https://openreview.net/pdf?id=JWUqwie0--W)
