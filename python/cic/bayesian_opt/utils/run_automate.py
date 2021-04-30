import argparse
import numpy as np
import os
import pybullet as p
import pickle as pkl
import subprocess
from subprocess import PIPE
import time
import json


def start_process(process_list,iteration,curr_run,res_path,initial_pos_path):
    #process_list.append(subprocess.Popen(['./launch_local.sh'], shell=True, cwd='/home/funk/Code/rrc_descartes/'))
    os.makedirs(str(res_path) + str(iteration) + '_output_' + str(curr_run))
    process_list.append(subprocess.Popen(['./run_in_simulation.py \
     --output-dir ' + str(res_path) + str(iteration) + '_output_' + str(curr_run) + '/ \
     --repository $(pwd) \
     --backend-image /home/funk/production.sif \
     --branch automate_bo \
     --nv \
     --initial-pos-file ' + str(initial_pos_path) + ' \
     --initial-pos-idx ' + str(curr_run)], shell=True, cwd='/home/funk/Code/rrc_descartes/'))
    return process_list



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Argument parser for BO framework')
    parser.add_argument('--path', type=str, required=True, help='where to store the results')

    args = parser.parse_args()

    if not os.path.exists(args.path):
        os.makedirs(args.path)
    else:
        print ("Output folder already exists -> FINISHING")
        exit()

    init_x = [-0.025, 0.025]
    init_y = [-0.025, 0.025]
    init_z = [0.0325, 0.0325]

    init_r = [0.0, 0.0]
    init_p = [0.0, 0.0]
    init_yaw = [0.0, 0.0] #[np.deg2rad(0.0), np.deg2rad(360.0)]

    target_x = [-0.05, 0.05]
    target_y = [-0.05, 0.05]
    target_z = [0.0325, 0.0325]

    target_r = [0.0, 0.0]
    target_p = [0.0, 0.0]
    target_yaw = [0.0, 0.0]

    num_runs = 2
    num_threads_available = 10
    iter = 0

    rand_arr = np.random.rand(12, num_runs)
    rand_arr = (rand_arr-0.5)
    rand_arr[0,:] = rand_arr[0,:]*(init_x[1]-init_x[0]) + 0.5*(init_x[1]+init_x[0])
    rand_arr[1,:] = rand_arr[1,:]*(init_y[1]-init_y[0]) + 0.5*(init_y[1]+init_y[0])
    rand_arr[2,:] = rand_arr[2,:]*(init_z[1]-init_z[0]) + 0.5*(init_z[1]+init_z[0])
    rand_arr[3,:] = rand_arr[3,:]*(init_r[1]-init_r[0]) + 0.5*(init_r[1]+init_r[0])
    rand_arr[4,:] = rand_arr[4,:]*(init_p[1]-init_p[0]) + 0.5*(init_p[1]+init_p[0])
    rand_arr[5,:] = rand_arr[5,:]*(init_yaw[1]-init_yaw[0]) + 0.5*(init_yaw[1]+init_yaw[0])
    rand_arr[6,:] = rand_arr[6,:]*(target_x[1]-target_x[0]) + 0.5*(target_x[1]+target_x[0])
    rand_arr[7,:] = rand_arr[7,:]*(target_y[1]-target_y[0]) + 0.5*(target_y[1]+target_y[0])
    rand_arr[8,:] = rand_arr[8,:]*(target_z[1]-target_z[0]) + 0.5*(target_z[1]+target_z[0])
    rand_arr[9,:] = rand_arr[9,:]*(target_r[1]-target_r[0]) + 0.5*(target_r[1]+target_r[0])
    rand_arr[10,:] = rand_arr[10,:]*(target_p[1]-target_p[0]) + 0.5*(target_p[1]+target_p[0])
    rand_arr[11,:] = rand_arr[11,:]*(target_yaw[1]-target_yaw[0]) + 0.5*(target_yaw[1]+target_yaw[0])

    target_arr = np.zeros((14,num_runs))
    for i in range(num_runs):
        target_arr[:3,i] = rand_arr[:3,i]
        target_arr[3:7,i] = np.asarray(p.getQuaternionFromEuler(rand_arr[3:6,i]))
        target_arr[7:10,i] = rand_arr[6:9,i]
        target_arr[10:14,i] = np.asarray(p.getQuaternionFromEuler(rand_arr[9:12,i]))

    initial_pos_path = str(args.path+'pos.pkl')
    with open(initial_pos_path, 'wb') as f:
        pkl.dump(target_arr,f)

    #subprocess.call(["cd ../..", "launch_local.sh"],shell=True)
    #subprocess.call(['./launch_local.sh'],shell=True, cwd='/home/funk/Code/rrc_descartes/')
    #subprocess.run(["cd ../..", "./launch_local.sh"],capture_output=True)

    process_list = []
    num_runs_completed = 0
    num_runs_started = 0

    while ((num_runs-num_runs_completed)>0):
        num_threads_open = np.clip(num_threads_available-len(process_list),0,num_runs-(num_runs_completed+len(process_list)))
        if (num_threads_open>0):
            for i in range(num_threads_open):
                process_list = start_process(process_list,iter,num_runs_started,args.path,initial_pos_path)
                num_runs_started += 1

        remove_list = []
        offset = 0
        for j in range(len(process_list)):
            if not (process_list[j].poll() is None):
                remove_list.append(j)
                num_runs_completed += 1
        for j in range(len(remove_list)):
            process_list.pop(remove_list[j]-offset)
            offset += 1

        print ("number of runs: ", num_runs)
        print ("number of runs completed: ", num_runs_completed)
        time.sleep(10)


    print ("Collect results")
    resulting_rewards = []
    for i in range(num_runs):
        file_path = args.path + str(iter) + '_output_' + str(i) + '/user_stdout.txt'
        fileHandle = open(file_path, "r")
        lineList = fileHandle.readlines()
        des_line = lineList[len(lineList)-2]
        des_line = des_line.split(":")
        resulting_rewards.append(float(des_line[-1]))

    with open(args.path + str(iter) + '_rew.txt', 'w') as f:
        f.write("Mean reward " + str(np.mean(np.asarray(resulting_rewards))) + '\n')
        f.write(json.dumps((np.asarray(resulting_rewards)).tolist()))


    print ("finish programme")
