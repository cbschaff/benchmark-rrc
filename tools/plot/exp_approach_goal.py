#!/usr/bin/env python3
'''
This code traverses a directories of evaluation log files and
record evaluation scores as well as plotting the results.
'''
import os
import argparse
import json
from shutil import copyfile
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utils import *

def _calc_pos_error(pos, goal_pos):
    xyz_dist = np.linalg.norm(
        np.asarray(pos) - np.asarray(goal_pos)
    )
    return xyz_dist


def generate_csv(log_dir, csv_file, df=None, tag=None, use_degree=True):
    '''
    Traverse and read log files, and then output csv file from the eval data.
    - file to be generated: 'eval_scores.csv'
    - columns: state_machine_id, timesteps, rot_error
    '''
    if df is None:
        df = pd.DataFrame(columns=['state_machine_id', 'reward'])
    assert 'state_machine_id' in df and 'reward' in df

    model_names = extract_model_names(log_dir)

    # Traverse all episodes and add each entry to data frame
    for state_machine_id, episode_idx, episode_dir in traverse_all_episodes(log_dir):
        reward_json = JsonUtil(os.path.join(episode_dir, 'reward.json'))
        goal_json = JsonUtil(os.path.join(episode_dir, 'goal.json'))
        obj_dropped = 'No' if goal_json.load('reachfinish') == -1 else 'Yes'

        goal_pos = goal_json.load(key='goal')['position']
        init_obj_pose = goal_json.load(key='init_obj_pose')['position']

        # NOTE: 'mp_with_cic_grasp' --> move_to_goal='mp', grasp='cic'
        move_to_goal, _, grasp, *_ = model_names[state_machine_id].split('_')
        entry = {
            'state_machine_id': state_machine_id,
            'state_machine_name': model_names[state_machine_id],
            'grasp': grasp,
            'move_to_goal': move_to_goal,
            'obj_dropped': obj_dropped,
            'init_pos_error': _calc_pos_error(goal_pos,init_obj_pose),
            'tag': tag,
            # 'goal_is_outside': True if episode_idx >= 20 else False,
            **goal_json.load(key=['init_align_obj_error', 'rot_error_final', 'pos_error_final']),
            **reward_json.load(key=['reward'])
        }

        # Convert radian --> degree
        if use_degree:
            entry['rot_error_final'] *= 180 / np.pi
            entry['init_align_obj_error'] *= 180 / np.pi

        df = df.append(entry, ignore_index=True)  # df.append works differently from python since it is stupid
    df.to_csv(csv_file, index=False)
    return df


def generate_plot(input_csv_file, plot_file):
    data = pd.read_csv(input_csv_file)
    print (data.groupby('state_machine_name').mean())


    # Create a 3-grasps x 3-move-to-goal plots
    sns.set_theme(style="ticks")
    g = sns.FacetGrid(
        data, col='grasp', row='move_to_goal', hue='obj_dropped',
        margin_titles=True,  # show col & row titles nicely
        despine=False  # show each plot in a box
    )
    # g.map(
    #     sns.scatterplot, 'init_align_obj_error', 'rot_error_final', style=data['goal_is_outside'], alpha=.7
    # )
    g.map_dataframe(
        sns.scatterplot, 'rot_error_final', 'pos_error_final', alpha=.7
    )
    # g.fig.subplots_adjust(wspace=0, hspace=0)
    g.add_legend()
    g.savefig(plot_file)
