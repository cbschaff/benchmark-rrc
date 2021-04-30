#!/usr/bin/env python3
import os
import argparse
import json
import copy
from shutil import copyfile


class JsonUtil:
    def __init__(self, filepath):
        assert filepath.endswith('.json')
        self.filepath = filepath
        self.dict_ = None
        with open(self.filepath, 'r') as f:
            self.dict_ = json.load(f)

    def load(self, key=None):
        if isinstance(key, str):
            return self.dict_.get(key)
        elif isinstance(key, list):
            return {k: v for k, v in self.dict_.items() if k in key}
        elif key is None:
            # Returns the entire dictionary
            return copy.deepcopy(self.dict_)
        else:
            raise ValueError()

    def add_and_save(self, key, val):
        assert self.dict_ is not None, 'call JsonUtil.load first'
        if not os.path.isfile(self.filepath + '.bak'):
            copyfile(self.filepath, self.filepath + '.bak')  # create a backup just in case
        with open(self.filepath, 'w') as f:
            self.dict_[key] = val
            json.dump(self.dict_, f)

def listdir(directory):
    '''similar to os.listdir but returns fullpath'''
    return [os.path.join(directory, file_) for file_ in os.listdir(directory)]


def traverse_all_episodes(log_dir):
    '''Traverses all episode directories.

    NOTE: expected file structure is as follows
    0
      0
        0_output_0
          goal.json ("align_roterror": int, "reachfinish": int  # self.reach_finish_point)
          ...
        0_output_1
          ...
    1
      0
        1_output_0
          ...
        1_output_1
          ...
    '''

    # only list directories whose names are decimal
    dirs = sorted([d for d in listdir(log_dir) if os.path.isdir(d) and d.split('/')[-1].isdecimal()])
    for state_machine_id, dir_ in enumerate(dirs):
        # dir_: RRC/0
        episode_dirs = sorted(listdir(os.path.join(dir_, '0')))
        for episode_idx, episode_dir in enumerate(episode_dirs):
            if (os.path.isdir(episode_dir)):
                yield (state_machine_id, episode_idx, episode_dir)


def extract_model_names(log_dir):
    with open(os.path.join(log_dir, 'MODEL_ID.txt'), "r") as f:
        json_str = str(f.read()).replace("'", '"')
    return json.loads(json_str)
