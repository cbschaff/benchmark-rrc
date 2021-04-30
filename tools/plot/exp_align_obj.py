#!/usr/bin/env python3
'''
This code traverses a directories of evaluation log files and
record evaluation scores as well as plotting the results.
'''
import os
import argparse
import json
import copy
from shutil import copyfile
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utils import *


MAX_ALIGN_STEPS = 75000 - 1  # This depends on the evaluation code used to generate the logs

def generate_csv(log_dir, csv_file):
    '''
    Traverse and read log files, and then output csv file from the eval data.
    - file to be generated: 'eval_scores.csv'
    - columns: state_machine_id, timesteps, rot_error
    '''
    df = pd.DataFrame(columns=['state_machine_id', 'state_machine_name', 'timesteps', 'rot_error'])
    model_names = extract_model_names(log_dir)

    # Traverse all episodes and add each entry to data frame
    for state_machine_id, episode_idx, episode_dir in traverse_all_episodes(log_dir):
        json_util = JsonUtil(os.path.join(episode_dir, 'goal.json'))
        entry = {
            'state_machine_id': state_machine_id,
            'state_machine_name': model_names[state_machine_id],
            **json_util.load()
        }

        # Handling the timesteps==-1 case
        if entry['reachfinish'] == -1:
            entry['reachfinish'] = MAX_ALIGN_STEPS

        if entry['reachstart'] == -1:
            raise ValueError('\'reachstart\' in {episode_dir}/goal.json does not contain a valid value.')

        # Rename dict keys
        entry['timesteps'] = entry.pop('reachfinish') - entry.pop('reachstart')
        entry['rot_error'] = entry.pop('align_obj_error')
        entry['init_rot_error'] = entry.pop('init_align_obj_error', None)

        # Add a new entry
        entry['rot_error_diff'] = entry['init_rot_error'] - entry['rot_error']

        df = df.append(entry, ignore_index=True)  # df.append works differently from python since it is stupid
    df.to_csv(csv_file, index=False)


def generate_plot(input_csv_file, plot_file):
    data = pd.read_csv(input_csv_file)
    sns.scatterplot(data=data, x="timesteps", y="rot_error", hue="state_machine_name", alpha=0.8)
    plt.savefig(plot_file)

