# Submitting and visualizing jobs run on the TriFinger robot cluster

This directory provides code for submitting jobs and analyzing logs by generating plots and videos.

### Account Verification

In order to use this code, you must first add your account information in two files placed in this directory:

1) user.txt: This should contain only the username and password (each on their own line) of your account

2) sshkey: This should contain the ssh private key used to log into the submission system

**Account information is provided by the competition organizers.**


### Submitting Jobs

To submit a job, edit the `roboch.json` configuration file to contain the proper information (see [this]( https://people.tuebingen.mpg.de/felixwidmaier/realrobotchallenge/robot_phase/submission_system.html#configuration-file-roboch-json)
for details).

Then simply calling `./submit_job.sh` will upload the `roboch.json` file and start a job on the cluster.
To run multiple, identical jobs in sequence, run `./submit_jobs_in_sequence.sh n`, where `n` is the number of jobs.

### Downloading and Analyzing Log Data

This code analyzes log data in the following ways:
- computes the accumulated reward of the episode
- generates a plot of desired vs actual joint positions of each finger of the TriFinger robot
- generates a video showing synchronized images from the 3 cameras observing the robot
- generates a video in which robot and object poses are rendered in pybullet and shown alongside camera images from the above video

Generating the videos requires ffmpeg to be installed with the libvpx codec, although another codec could be used in its place.

To generate the above files for all submitted jobs, run the following command:
```bash
./run.sh /path/to/log/directory /path/to/singularity/image.sif
```

If called multiple times, the `run.sh` script skips jobs that have already been processed.
