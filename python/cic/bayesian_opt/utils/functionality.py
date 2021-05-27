import argparse
import numpy as np
import os
import pybullet as p
import pickle as pkl
import subprocess
from subprocess import PIPE
import time
import json
import shutil

import sys
sys.path.append("../")
from const import SIMULATION, GITHUB_BRANCH, DIFFICULTY_LEVEL, EVALUATE_TRAJ, PATH_TO_IMAGE, USERNAME, PWD, NUM_LOCAL_THREADS, EPISODE_LEN_REAL

def modify_and_push_json():
    with open("./utils/roboch.json", "r") as jsonFile:
        data = json.load(jsonFile)

    data["branch"] = GITHUB_BRANCH
    data["episode_length"] = EPISODE_LEN_REAL

    with open("./utils/roboch.json", "w") as jsonFile:
        json.dump(data, jsonFile, indent=2)

    if (SIMULATION):
        print("Uploading roboch file not needed")
    else:
        task = subprocess.Popen(['./upload_json.sh \
      ' + USERNAME + ' ' + PWD], shell=True, cwd='./utils/')
        while True:
            if not (task.poll() is None):
                break
            else:
                time.sleep(0.5)

def select_robot_json(robot):
    with open("./utils/roboch.json", "r") as jsonFile:
        data = json.load(jsonFile)

    data["branch"] = GITHUB_BRANCH
    data["episode_length"] = EPISODE_LEN_REAL
    if (robot is None):
        data["group"] = "USER"
    else:
        data["group"] = robot

    with open("./utils/roboch.json", "w") as jsonFile:
        json.dump(data, jsonFile, indent=2)


def push_github(command):
    if (SIMULATION):
        print ("Pushing not needed since we run in simulation")
    else:
        task = subprocess.Popen([("./push_github.sh" + " " + str(command) + " " + str(GITHUB_BRANCH))], shell=True, cwd='./utils/')
        while True:
            if not (task.poll() is None):
                break
            else:
                time.sleep(0.5)

def start_process(process_list,iteration,curr_run,res_path,initial_pos_path, params_path):
    #process_list.append(subprocess.Popen(['./launch_local.sh'], shell=True, cwd='/home/funk/Code/rrc_descartes/'))
    dir = str(res_path) + str(iteration) + '_output_' + str(curr_run)

    with open(('./content/iter_idx.txt'), 'w') as f:
        f.write(str(curr_run))

    if os.path.exists(dir):
        shutil.rmtree(dir)

    process_list.append(subprocess.Popen(['./run_locally_bo.sh ' + PATH_TO_IMAGE + ' ros2 run rrc run_local_episode_bo.py' + ' ' + str(DIFFICULTY_LEVEL) + ' ' +str(dir)], shell=True, cwd='../../../'))

    while not(os.path.exists(dir)):
        print ("Directory not yet created -> wait")
        time.sleep(1)
    # wait one additional second until has started
    print ("JOB HAS STARTED")

    return process_list

def start_process_real(process_list,iteration,curr_run,res_path,initial_pos_path, params_path):
    #process_list.append(subprocess.Popen(['./launch_local.sh'], shell=True, cwd='/home/funk/Code/rrc_descartes/'))
    dir = str(res_path) + str(iteration) + '_output_' + str(curr_run)
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir)
    with open(('./content/iter_idx.txt'), 'w') as f:
        f.write(str(curr_run))  #"denotes the iteration concerning start position"
        #f.write('\n')
        #f.write(str(iteration))      # "with respect to parameter file,.."
    push_github("modifying_index")
    process_list.append(subprocess.Popen(['./automated_submission_real.sh \
      ' + str(res_path) + str(iteration) + '_output_' + str(curr_run) + '/' \
     + ' ' + USERNAME + ' ' + PWD], shell=True, cwd='./utils/'))
    return process_list


def run_param_rollout(iter, initial_pos_path, general_path, params_path,run_eval, outside_idx=None):

    if (run_eval):
        with open(initial_pos_path, 'rb') as f:
            arr = pkl.load(f)

        if (outside_idx is None):
            num_runs = np.shape(arr)[1]
        else:
            num_runs = 1

        num_threads_available = NUM_LOCAL_THREADS#5

        process_list = []
        idx_runs_started = []
        idx_runs_missing = []
        num_runs_completed = 0
        if (outside_idx is None):
            num_runs_started = 0
        else:
            num_runs_started = outside_idx

        while ((num_runs - num_runs_completed) > 0):
            num_threads_open = np.clip(num_threads_available - len(process_list), 0,
                                       num_runs - (num_runs_completed + len(process_list)))
            if (num_threads_open > 0):
                for i in range(num_threads_open):
                    if (len(idx_runs_missing)==0):
                        process_list = start_process(process_list, iter, num_runs_started, general_path, initial_pos_path, params_path)
                        idx_runs_started.append(num_runs_started)
                        num_runs_started += 1
                    else:
                        process_list = start_process(process_list, iter, idx_runs_missing[0], general_path,
                                                     initial_pos_path, params_path)
                        idx_runs_started.append(idx_runs_missing[0])
                        idx_runs_missing.pop(0)

            remove_list = []
            offset = 0
            for j in range(len(process_list)):
                if not (process_list[j].poll() is None):
                    finish_status, rew = check(str(general_path) + str(iter) + '_output_' + str(idx_runs_started[j]))

                    if (finish_status):
                        num_runs_completed += 1
                    else:
                        print ("RUN NOT FINISHED -> STARTING AGAIN")
                        idx_runs_missing.append(idx_runs_started[j])
                    # TODO: since we run on the real system we MUST check whether success or not,...
                    remove_list.append(j)

            for j in range(len(remove_list)):
                process_list.pop(remove_list[j] - offset)
                idx_runs_started.pop(remove_list[j] - offset)
                offset += 1

            print("number of runs: ", num_runs)
            print("number of runs completed: ", num_runs_completed)
            time.sleep(10)

    print("Collect results")
    resulting_rewards = []
    if (outside_idx is None):
        for i in range(num_runs):
            specific_path = general_path + str(iter) + '_output_' + str(i) + '/'
            # use the specified function to compute the reward -> scales to sim and real system,...
            task = subprocess.Popen([(PATH_TO_IMAGE + " ros2 run robot_fingers evaluate_trajectory.py" " " + str(specific_path) + " " + str(int(SIMULATION)) + " " + EVALUATE_TRAJ )], shell=True, cwd='./utils/')
            while True:
                if not (task.poll() is None):
                    break
                else:
                    time.sleep(0.5)

            with open(os.path.join(specific_path, 'reward.json'), 'r') as f:
                info = json.load(f)

            # print (info['reward'])
            # print (info['valid'])
            # input ("WAIT")

            # # rew = evaluate_trajectory.evaluate_trajectory(SIMULATION, general_path + str(iter) + '_output_' + str(i), EVALUATE_TRAJ)
            # file_path = general_path + str(iter) + '_output_' + str(i) + '/reward.pkl'
            # with open(file_path, 'rb') as f:
            #     arr = pkl.load(f)
            # arr = np.asarray(arr)
            # rew = np.sum(arr)
            # print (rew)
            # in simulation: result is "always valid,..."
            resulting_rewards.append(info['reward'])
    else:
        i = outside_idx
        specific_path = general_path + str(iter) + '_output_' + str(i) + '/'
        # use the specified function to compute the reward -> scales to sim and real system,...
        task = subprocess.Popen([(PATH_TO_IMAGE + " ros2 run robot_fingers evaluate_trajectory.py" " " + str(
            specific_path) + " " + str(int(SIMULATION)) + " " + EVALUATE_TRAJ)], shell=True, cwd='./utils/')
        while True:
            if not (task.poll() is None):
                break
            else:
                time.sleep(0.5)

        with open(os.path.join(specific_path, 'reward.json'), 'r') as f:
            info = json.load(f)

        resulting_rewards.append(info['reward'])


    with open(general_path + str(iter) + '_rew.txt', 'a') as f:
        f.write("Mean reward " + str(np.mean(np.asarray(resulting_rewards))) + '\n')
        f.write(json.dumps((np.asarray(resulting_rewards)).tolist()))


    print("finish programme")
    return (np.mean(np.asarray(resulting_rewards)))

def check(path):
    specific_path = path + '/'

    file_path = specific_path + "goal.json"

    #check if all files are there:
    if not(SIMULATION):
        are_there = os.path.exists(specific_path+"goal.json") and os.path.exists(specific_path+"camera_data.dat") and os.path.exists(specific_path+"robot_data.dat")
    else:
        are_there = True

    if (are_there):
        try:
            with open(file_path) as f:
                print("I will do some Magic with this")


                # use the specified function to compute the reward -> scales to sim and real system,...
                task = subprocess.Popen([(PATH_TO_IMAGE + " ros2 run robot_fingers evaluate_trajectory.py" " " + str(
                    specific_path) + " " + str(int(SIMULATION)) + " " + EVALUATE_TRAJ)], shell=True, cwd='./utils/')
                while True:
                    if not (task.poll() is None):
                        break
                    else:
                        time.sleep(0.5)

        except FileNotFoundError:
            print("File not fond -> TIMEOUT")
            time.sleep(10)
            return False, -10000

    else:
        print("Files not found -> TIMEOUT")
        time.sleep(10)
        return False, -10000

    try:
        with open(os.path.join(specific_path, 'reward.json'), 'r') as f:
            info = json.load(f)
            # print (info['reward'])
            # print (info['valid'])
    except FileNotFoundError:
        print("File not fond -> TIMEOUT")
        time.sleep(10)
        return False, -10000

    return info['valid'], info['reward']


def run_param_rollout_real(iter, initial_pos_path, general_path, params_path, run_eval, outside_idx=None):

    if (run_eval):
        with open(initial_pos_path, 'rb') as f:
            arr = pkl.load(f)

        if (outside_idx is None):
            num_runs = np.shape(arr)[1]
        else:
            num_runs = 1

        num_threads_available = 1

        process_list = []
        idx_runs_started = []
        idx_runs_missing = []
        num_runs_completed = 0
        if (outside_idx is None):
            num_runs_started = 0
        else:
            num_runs_started = outside_idx

        resulting_rewards = []

        while ((num_runs - num_runs_completed) > 0):
            num_threads_open = np.clip(num_threads_available - len(process_list), 0,
                                       num_runs - (num_runs_completed + len(process_list)))
            if (num_threads_open > 0):
                for i in range(num_threads_open):
                    if (len(idx_runs_missing) == 0):
                        process_list = start_process_real(process_list, iter, num_runs_started, general_path, initial_pos_path, params_path)
                        idx_runs_started.append(num_runs_started)
                        num_runs_started += 1
                    else:
                        process_list = start_process_real(process_list, iter, idx_runs_missing[0], general_path, initial_pos_path, params_path)
                        idx_runs_started.append(idx_runs_missing[0])
                        idx_runs_missing.pop(0)

            remove_list = []
            offset = 0
            for j in range(len(process_list)):
                if not (process_list[j].poll() is None):
                    finish_status, rew = check(str(general_path) + str(iter) + '_output_' + str(num_runs_started-1))

                    if (finish_status):
                        resulting_rewards.append(rew)
                        num_runs_completed += 1
                    else:
                        idx_runs_missing.append(idx_runs_started[j])
                    # TODO: since we run on the real system we MUST check whether success or not,...
                    remove_list.append(j)

            for j in range(len(remove_list)):
                process_list.pop(remove_list[j] - offset)
                idx_runs_started.pop(remove_list[j] - offset)
                offset += 1

            print("number of runs: ", num_runs)
            print("number of runs completed: ", num_runs_completed)
            time.sleep(10)
    else:
        with open(initial_pos_path, 'rb') as f:
            arr = pkl.load(f)
        resulting_rewards = []
        num_runs = np.shape(arr)[1]
        num_runs_completed = 0
        num_runs_started = 0
        while ((num_runs - num_runs_completed) > 0):
            num_runs_started += 1
            finish_status, rew = check(str(general_path) + str(iter) + '_output_' + str(num_runs_started - 1))
            if (finish_status):
                resulting_rewards.append(rew)
                num_runs_completed += 1




    with open(general_path + str(iter) + '_rew.txt', 'a') as f:
        f.write("Mean reward " + str(np.mean(np.asarray(resulting_rewards))) + '\n')
        f.write(json.dumps((np.asarray(resulting_rewards)).tolist()))


    print("finish programme")
    return (np.mean(np.asarray(resulting_rewards)))
