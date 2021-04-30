"""
Bayesian Optimization experiment runner.

Relies heavily on BoTorch.
"""

import os
import logging
import matplotlib.pyplot as plt
import numpy as np
import torch
import random
import sys
# sys.path.append("../")
import pickle as pkl

from tqdm import tqdm

import shutil
from distutils.spawn import find_executable

from utils.functionality import run_param_rollout_real, run_param_rollout
from utils.functionality import push_github, modify_and_push_json, select_robot_json
from utils.sampling_functions import define_sample_fct

from const import SIMULATION, GITHUB_BRANCH, DIFFICULTY_LEVEL, SAMPLE_FCT, NUM_INIT_SAMPLES, NUM_ROLLOUTS_PER_SAMPLE, NUM_ITERATIONS, NUM_ACQ_RESTARTS, ACQ_SAMPLES
from const import SAMPLE_NEW, MODELS_TO_RUN, ROBOTS_AVAILABLE, SPLIT_JOBS_EVENLY
from utils import normalization_tools

logger = logging.getLogger(__file__)

def plot_uncertainty(ax, x, mean, variance, stds=[2], color='b', alpha=0.1):
    x = x.squeeze()
    mean = mean.squeeze()
    variance = variance.squeeze()
    for sf in stds:
        y_std = sf * np.sqrt(variance)
        y_upper = mean + y_std
        y_lower = mean - y_std
        ax.fill_between(x, y_upper, y_lower, where=y_upper>y_lower, color=color, alpha=alpha)


# Constants

DIR_NAME = os.path.dirname(__file__)
# NOT WORKING PROPERLY AT THE MOMENT
#TORCH_DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
TORCH_DEVICE = torch.device("cpu")
print(f"Using {TORCH_DEVICE}")


# Tasks

# Helpers

def to_tensor(x):
    return torch.from_numpy(x).float()

def update_data(idx, x_train, y_train, x_bank, y_bank):
    indexes = [*range(x_bank.shape[0])]
    indexes.pop(idx)
    x_train = torch.cat((x_train, x_bank[idx].view(1,1)), 0)
    y_train = torch.cat((y_train, y_bank[idx].view(1,1)), 0)
    x_bank = x_bank[indexes]
    y_bank = y_bank[indexes]
    return x_train, y_train, x_bank, y_bank

def save_metrics(y_history, y_opt, y_opt_est, res_dir):
    ns, ni = y_history.shape
    regret = y_opt * np.ones((ns, ni)) - y_history
    log_regret = np.log(regret)
    metrics = {
        "y_history": y_history,
        "regret": regret,
        "log_regret": log_regret,
        "y_opt_est": y_opt_est,
    }
    np.savez_compressed(os.path.join(res_dir, 'metrics.npz'), **metrics)

    x = [*range(ni)]
    p25, p75 = np.percentile(y_history, (25, 75), axis=0)
    f, a = plt.subplots(1, 1)
    a.fill_between(x, p25, p75, where=p75>p25, color='b', alpha=0.2)
    for i in range(ns):
        a.plot(y_history[i, :], 'b.--', alpha=0.5)
    a.plot([y_opt] * ni, 'k--')
    a.set_xlabel("Iterations")
    a.set_ylabel("$y$")
    plt.savefig(os.path.join(res_dir, f"metric.png"),
        bbox_inches='tight', format='png')
    plt.close(f)


    p25, p50, p75 = np.percentile(regret, (25, 50, 75), axis=0)
    f, a = plt.subplots(1, 1)
    a.fill_between(x, p25, p75, where=p75>p25, color='b', alpha=0.2)
    for i in range(ns):
        a.plot(regret[i, :], 'b.--', alpha=0.5)
    a.plot(p50, 'b')
    a.set_yscale('log')
    a.set_xlabel("Iterations")
    a.set_ylabel("Immediate Regret")
    plt.savefig(os.path.join(res_dir, f"regret.png"),
        bbox_inches='tight', format='png')
    plt.close(f)



class RRC_v1(object):
    """Sinc in a haystack task."""
    # number of initial random points
    num_init_samples = NUM_INIT_SAMPLES #1#10
    # number of BO updates
    num_iter = NUM_ITERATIONS #50
    # number of restarts for optimizing the acquisition function
    num_acq_restarts = NUM_ACQ_RESTARTS#100
    # number of index_set for used for optimizing the acquisition function
    num_acq_samples = ACQ_SAMPLES#500

    plot_model = True
    # d_x should be dimension of x,..
    d_x = 1
    x_min = np.array([0.0])
    x_max = np.array([0.02])
    #TODO: identify meaning of y_opt, x_opt. Is this initial guess?
    y_opt = 0
    x_opt = np.array([0.0])

    param_normalizer = normalization_tools.UnitCubeProjector(x_min,x_max)

    @staticmethod
    def function(x,path,globcount,run_eval, index):

        if True:
            # create directory for this run
            if (run_eval):
                if not(os.path.exists((path)+str(globcount))):
                    os.makedirs((path)+str(globcount))
            else:
                input("Copy files now,..")

            result = np.zeros((np.shape(x)[0],1))
            # for very parameter run the evaluation,...
            for i in range(np.shape(x)[0]):
                path_curr_params = (path) + str(globcount) + '/' + str(i) + '/'
                if (run_eval):
                    if not(os.path.exists(path_curr_params)):
                        os.makedirs(path_curr_params)
                    # store the current parameters in pickle file and add to our current path
                    with open((path_curr_params+'params.pkl'),'wb') as f:
                        pkl.dump(np.asarray(x[i,:]),f)
                    # store the current parameters in pickle file and add to repo
                    with open(('./content/params.pkl'),'wb') as f:
                        pkl.dump(np.asarray(x[i,:]),f)
                    # push the parameters,...
                    push_github("pushing_current_params")

                # now collect the results on the real system,...
                if (SIMULATION):
                    result[i,:] = run_param_rollout(globcount, str(path+'pos.pkl'), path_curr_params, str(path_curr_params+'params.pkl'), run_eval, index)
                else:
                    result[i,:] = run_param_rollout_real(globcount, str(path+'pos.pkl'), path_curr_params, str(path_curr_params+'params.pkl'), run_eval, index)

            return result

    def __call__(self, x, path, globcount=None, run_eval=True, index=None):
        #noise_std = 0.
        return self.function(x,path,globcount,run_eval, index) #+ noise_std * np.random.randn(*x.shape)

    def x_sample(self, n=1):
        return np.random.uniform(self.x_min, self.x_max, (n, self.d_x)).reshape((n, self.d_x))





EXPERIMENTS = {
    "rrc_v1"    : RRC_v1,
}

def main(args, res_dir):
    modify_and_push_json()
    # TODO: potentially create outer loop to make sure to more randomize the experiments,....


    number_rollouts_per_param = NUM_ROLLOUTS_PER_SAMPLE
    # Sample potential goal and target positions.
    # WRITE DIFFICULTY LEVEL TO FILE
    with open(('./content/diff.txt'), 'w') as f:
        f.write(str(DIFFICULTY_LEVEL))

    # ONLY SAMPlE IF DESIRED
    if (SAMPLE_NEW):
        define_sample_fct(SAMPLE_FCT, number_rollouts_per_param, ("./content"+'/'), path2=(res_dir+'/'))
    push_github("push_position_file")
    #input ("WAIT")
    logging_txt_file = (res_dir+'/' + "RESULTS_LOGGING.txt")
    model_id_file = (res_dir + '/' + "MODEL_ID.txt")
    with open(model_id_file,"w") as f:
        f.write(str(MODELS_TO_RUN))

    # Generate list which contains model_id, goal_idx of all experiments that have to be run
    if (SPLIT_JOBS_EVENLY):
        split_index = 0
        split_counter = int(round(number_rollouts_per_param / len(ROBOTS_AVAILABLE)))
    combination_to_be_run = []
    for j in range(number_rollouts_per_param):
        if (SPLIT_JOBS_EVENLY):
            if (j!=0 and j%split_counter==0):
                split_index += 1
                split_index = np.clip(split_index,0,len(ROBOTS_AVAILABLE)-1)
        for i in range((len(MODELS_TO_RUN))):
            if (SPLIT_JOBS_EVENLY):
                combination_to_be_run.append([i, j, ROBOTS_AVAILABLE[split_index]])
            else:
                combination_to_be_run.append([i,j,random.choice(ROBOTS_AVAILABLE)])

    print (combination_to_be_run)
    input ("WAIT")
    random.shuffle(combination_to_be_run)
    random.shuffle(combination_to_be_run)
    random.shuffle(combination_to_be_run)
    random.shuffle(combination_to_be_run)
    random.shuffle(combination_to_be_run)
    order_exec_file = (res_dir + '/' + "ORDER_OF_EXECUTION.txt")
    # log in which order the experiments are run
    with open(order_exec_file,"w") as f:
        f.write(str(combination_to_be_run))



    reference_path = (res_dir+'/')

    logger.info("Args:")
    for k, v in vars(args).items():
        logger.info("%s: %s", k, v)

    task = RRC_v1()


    globcount_thres = -1


    with open(logging_txt_file,"w") as f:
        f.write("SEED: " + str(args.seed) + '\n')

    globcount = 0

    # Generate initial data
    _pt = task.x_sample(1)


    if (globcount<=globcount_thres):
        _y_train_raw = task(_pt, reference_path, globcount,run_eval=False)
    else:
        # determine model to run
        curr_combination = combination_to_be_run.pop()
        with open(('./content/model.txt'), 'w') as f:
            f.write(str(MODELS_TO_RUN[curr_combination[0]]))
        push_github("push_model_file")
        select_robot_json(curr_combination[2])
        modify_and_push_json()

        _y_train_raw = task(_pt, reference_path, curr_combination[0], run_eval=True,index=curr_combination[1])

    globcount += 1


    # Get best observed value from dataset
    with open(logging_txt_file,"a") as f:
        f.write("iteration: " + str(globcount) + '\n')
        f.write("val: " + str(_y_train_raw.item()) + '\n')


    # Bayesian Optimization Loop
    num_iter = len(combination_to_be_run)
    for i in tqdm(range(num_iter), total=num_iter):
        _pt = task.x_sample(1)
        if (globcount <= globcount_thres):
            y_new = task(_pt, reference_path, globcount, run_eval=False)
        else:
            curr_combination = combination_to_be_run.pop()
            with open(('./content/model.txt'), 'w') as f:
                f.write(str(MODELS_TO_RUN[curr_combination[0]]))
            push_github("push_model_file")
            select_robot_json(curr_combination[2])
            modify_and_push_json()

            y_new = task(_pt, reference_path, curr_combination[0], run_eval=True, index=curr_combination[1])
        globcount += 1


        with open(logging_txt_file, "a") as f:
            f.write("iteration: " + str(globcount) + '\n')
            f.write("val: " + str(y_new.item()) + '\n')

    print ("PROGRAMME FINISHED,...")




if __name__ == "__main__":
    import argparse

    from datetime import datetime

    DATETIME = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


    def make_results_folder(name, datetime=True, abs_path=''):
        _name = name.replace(" ", "_")
        folder = f"{DATETIME}_{_name}" if datetime else _name
        res_dir = os.path.join(abs_path, "_results", folder)
        if not os.path.exists(res_dir):
            os.makedirs(res_dir)
        elif datetime:
            shutil.rmtree(res_dir)
            os.makedirs(res_dir)
        print(f"Results folder: {res_dir}")
        return res_dir


    def configure_matplotlib():
        import matplotlib
        matplotlib.rcParams['font.family'] = 'serif'
        matplotlib.rcParams['figure.figsize'] = [19, 19]
        matplotlib.rcParams['legend.fontsize'] = 16
        matplotlib.rcParams['axes.titlesize'] = 22
        matplotlib.rcParams['axes.labelsize'] = 22
        if find_executable("latex"):
            matplotlib.rcParams['text.usetex'] = True


    def setup_logger(logger, res_dir, level=logging.INFO):
        hdlr = logging.FileHandler(os.path.join(res_dir, "output.log"))
        formatter = logging.Formatter('[%(asctime)s] %(pathname)s:%(lineno)d %(levelname)s - %(message)s','%d-%m %H:%M:%S')
        hdlr.setFormatter(formatter)
        logger.addHandler(hdlr) 
        logger.setLevel(level)


    parser = argparse.ArgumentParser(description="Run Experiment")
    parser.add_argument("--experiment", help="Task to run", default="DefaultExp")
    parser.add_argument("--path", help="Path where results are to be stored", default="")
    parser.add_argument("-s", "--seed", type=int, help="Random seed", default=0)

    args = parser.parse_args()


    name = str(args.experiment)
    res_dir = make_results_folder(name, datetime=True, abs_path = args.path)

    configure_matplotlib()
    setup_logger(logger, res_dir)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    try:
        main(args, res_dir)
    except:
        logger.exception("Experiment failed:")
        raise
    plt.show()
