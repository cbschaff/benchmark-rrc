# What is this?

A set of scripts to generate plot for align-obj experiments and mix-and-match experiments.  
This requires logs generated from [TUDarmstadt's evaluation code](https://github.com/cbschaff/real-robot-challenge-collab/blob/main/python/cic/bayesian_opt/eval_code.py).


# How to use this:

1. Make sure that you have a properly structured log directory tree (the format is described in docstring of `traverse_all_episodes` function in `utils.py`)
2. For align-obj experiments, run following:
 ```bash
 python3 plot.py --exp align --log-dir /path/to/log/directory
 ```
 NOTE: Make sure that you have matplotlib, seaborn and pandas.  
 You find the scatter plot at `plot.pdf` and raw data entries at `eval_scores.csv`

3. For mix-and-match experiments, run following:
 ```bash
 python3 plot.py --exp mix-match --log-dir /path/to/log/directory
 ```
 NOTE: Make sure that you have matplotlib, seaborn and pandas.  
 You find the scatter plot at `plot.pdf` and raw data entries at `eval_scores.csv`
