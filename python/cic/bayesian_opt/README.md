This part of the repository contains all of the functionality to run the bayesian optimization for optimizing the
hyperparameters.

The optimization procedure can be run both - in simulation - as well as on the real system. The most important steps
to run the procedure are summarized in the following and split into actions that have to be performed once and actions
to configure which configuration should be run.

# Actions to be executed once:

## 1) Creating a python environment for the experiments

The code has several dependencies which can be installed by reproducing the conda environment placed inside 'utils/env.yml'.

Additionally, we use normalization tools from the following repository: https://github.com/famura/SimuRLacra. To install
the required dependencies, it is sufficient to navigate into the 'Pyrado' folder and to simply install via: 'pip install
-e .'.

## 2) Adapting the singularity file

As the BO makes use of its "own" evaluation procedure, it is necessary to adapt
the standard image from phase 2 by using the following commands:

First "unpack" the image from Phase 2:
```
singularity build --sandbox lvl2/ realrobotchallenge_phase2.sif
```
Then the new evaluation file has to be placed inside of it, i.e. move the file 'utils/evaluate_trajectory.py' (which is
from this repository) into 'lvl2/opt/blmc_ei/src/robot_fingers/scripts/'.

Make sure that the file is executeable by performing:
```
chmod +x evaluate_trajectory.py
```
Then build a new container via
```
sudo singularity build production_test.sif lvl2/
```
As we also need gin inside the image, lastly use the script 'utils/adapt_image.def' and execute
```
sudo singularity build production_test1.sif adapt_image.def
```
For this operation it is required that it is executed with also the previously built file "production_test.sif" on the
same level.

## 3) Adapting paths and copy training scripts

- put the path of the freshly built singularity image into the 'const.py' file by setting the PATH_TO_IMAGE variable
- copy the two files 'run_episode_bo.py' and 'run_local_episode_bo.py' from the folder 'utils' into the already existing 
folder 'scripts'. The 'scripts' folder is located on the top-level of the repository. Note: In both of the files, first
the parameters to be optimized are set and then the experiments are launched.

## 4) (Optional) Only if to be run on the real system:
- specify the github branch which contains the code inside the 'const.py' file (Variable GITHUB_BRANCH)
- specify your credentials to submit jobs on the real system (Variables USERNAME and PWD)
- make sure that the run script ('run') (on the top level of the repo) contains the following command: 'ros2 run rrc run_episode_bo.py "$@"'
(i.e. you can also simply rename the file 'run_real_bo' to 'run' as this one already contains the right command)

# Actions to be executed before each experiment:
## General choices
- specify whether to run in simulation or on the real system by setting the SIMULATION variable inside const.py
- specify the difficulty level by setting the DIFFICULTY_LEVEL variable inside const.py
- specify which evaluation function to use by setting the EVALUATE_TRAJ variable inside const.py . This function
determines the objective function which is needed to calculate the value for each rollout
- specify the sampling function by setting SAMPLE_FCT inside const.py . The sampling function specifies both, the starting
positions of the object (only important when running in simulation) as well as the goal positions that should be reached
- specify whether to use the relative movement functionality by setting RELATIVE_MOVEMENT inside const.py . If set to
true, then the goal position is calculated relative to the current position of the object.
- if running in simulation specify how many threads are available, i.e. how many experiments you want to run in parallel
by setting the NUM_LOCAL_THREADS variable
## Choices specific for the bayesian optimization algorithm
- NUM_INIT_SAMPLES variable determines with how many initial samples the BO algorithm starts
- NUM_ROLLOUTS_PER_SAMPLE variable specifies how many experiments per choice of parameters are executed. This is especially
of importance when running experiments on the real system as for instance the initial starting position of the object is
not the same for each experiment which obviously influences the performance. Therefore we currently average the result
over NUM_ROLLOUTS_PER_SAMPLE runs
- NUM_ITERATIONS specifies for how many iterations we run the BO procedure
- NUM_ACQ_RESTARTS determines how often we restart the aquisition function and ACQ_SAMPLES tells how many samples we
draw from the aquisiton function

## Specifying the parameters:
- on the one hand the parameters, their lower and upper bound have to be specified inside 'bayes_opt_main.py' (lines 223-228)
- you also have to ensure that they are properly loaded by the scripts that are executed, i.e. by 'run_episode_bo.py' / 'run_local_episode_bo.py'.

# Launching the experiment:

Finally, the experiment can be launched via:
```
python bayes_opt_main.py rrc_v1 bo --path PATH_TO_RESULTS_FOLDER
```

Remarks: 
* Remember to execute this command from the activated conda environment.
* If you performed all the steps under **Actions to be executed once** then the above command will be functional
and launch a BO experiment in simulation (No further adaptions are required)
