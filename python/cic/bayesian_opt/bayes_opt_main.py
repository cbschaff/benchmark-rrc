"""
Bayesian Optimization experiment runner.

Relies heavily on BoTorch.
"""

import os
import logging
import matplotlib.pyplot as plt
import numpy as np
import torch
import sys
# sys.path.append("../")
import pickle as pkl

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.acquisition import ExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.constraints import GreaterThan

from botorch.models.model import Model as BotorchModel

from tqdm import tqdm

from torch.distributions import Normal

import shutil
from distutils.spawn import find_executable

from utils.functionality import run_param_rollout_real, run_param_rollout
from utils.functionality import push_github, modify_and_push_json
from utils.sampling_functions import define_sample_fct

from const import SIMULATION, GITHUB_BRANCH, DIFFICULTY_LEVEL, SAMPLE_FCT, NUM_INIT_SAMPLES, NUM_ROLLOUTS_PER_SAMPLE, NUM_ITERATIONS, NUM_ACQ_RESTARTS, ACQ_SAMPLES
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

# MODELS

class GaussianProcess(object):

    def __init__(self, dx, param_normalizer, *args, **kwargs):
        print (dx)
        self.param_normalizer = param_normalizer
        self.data_normalizer = normalization_tools.Standardizer()
        self.gp = None

    def fit(self, x_train, y_train):
        # normalize parameter (=input) data
        x_train_norm = self.param_normalizer.project_to(x_train)
        # normalize the data
        y_train_norm = self.data_normalizer.standardize(y_train)

        self.gp = SingleTaskGP(x_train_norm, y_train_norm)
        self.gp.likelihood.noise_covar.register_constraint("raw_noise", GreaterThan(1e-5))
        mll = ExactMarginalLogLikelihood(self.gp.likelihood, self.gp)
        fit_gpytorch_model(mll)
        return self.gp

    def predict(self, x):
        x_norm = self.param_normalizer.project_to(x)
        self.gp.eval()
        self.gp.likelihood.eval()
        with torch.set_grad_enabled(False):
            pred = self.gp(x_norm)
        return self.data_normalizer.unstandardize(pred.mean.view(-1, 1)), self.data_normalizer.unstandardize_wo_mean(pred.variance)


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


def plot_model(model, x, y, utility, x_train, y_train, x_new, y_new, utility_new, y_history, y_opt, name=""):
    posterior = model(x)
    mean = posterior.mean
    variance = posterior.variance

    f, a = plt.subplots(2, 1, sharex=True)
    a[0].plot(x, y_opt * np.ones(x.shape), 'c--')
    a[0].plot(x, y, 'y')
    a[0].plot(x_train, y_train, 'gx',  markersize=10, label="train")
    a[0].autoscale(axis='y')
    a[0].autoscale(False)
    if x_new is not None:
        a[0].plot(x_new, y_new, 'r*', markersize=10, label="selected")
    a[0].plot(x, mean, 'b--', label="model")
    plot_uncertainty(a[0], x.squeeze(), mean, variance, stds=[3, 2, 1])
    a[0].set_ylabel("y")
    a[0].legend()

    a[1].plot(x, utility, 'k')
    if x_new is not None:
        a[1].plot(x_new, utility_new, 'r*', markersize=10, label="selected")
    a[1].legend(loc='upper right')
    a[1].set_ylabel("Acquisition")

    a[-1].set_xlabel("x")
    plt.savefig(os.path.join(res_dir, f"model_{name}.png"),
        bbox_inches='tight', format='png')
    plt.close(f)

class Toy(object):
    """Sinc in a haystack task."""
    # number of initial random points
    num_init_samples = NUM_INIT_SAMPLES#4
    # number of BO updates
    num_iter = NUM_ITERATIONS #40
    # number of restarts for optimizing the acquisition function
    num_acq_restarts = NUM_ACQ_RESTARTS#100
    # number of index_set for used for optimizing the acquisition function
    num_acq_samples = ACQ_SAMPLES#500

    plot_model = True
    # d_x should be dimension of x,..
    d_x = 1
    x_min = np.array([-3])
    x_max = np.array([3])
    y_opt = 1
    x_opt = np.array([-1])

    @staticmethod
    def function(x):
        x = x + 1
        return np.sinc(6 * x)

    def __call__(self, x):
        noise_std = 0.
        return self.function(x) + noise_std * np.random.randn(*x.shape)

    def x_sample(self, n=1):
        return np.random.uniform(self.x_min, self.x_max, (n, self.d_x)).reshape((n, self.d_x))

    def x_domain(self, n=1):
        return np.linspace(self.x_min, self.x_max, n).reshape((n, self.d_x))

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
    d_x = 9
    x_min = np.array([-0.02, -0.02, 500, 800, 0.0075,0.25,0.001, 0.005, 0.005])
    x_max = np.array([0.02, 0.02, 1200, 1500, 0.035, 0.6,0.01, 0.04, 0.04])
    #TODO: identify meaning of y_opt, x_opt. Is this initial guess?
    y_opt = 0
    x_opt = np.array([0.0, 0.0, 700, 1350, 0.02, 0.5, 0.001, 0.01, 0.01])

    param_normalizer = normalization_tools.UnitCubeProjector(x_min,x_max)

    @staticmethod
    def function(x,path,globcount,run_eval):
        if True:
            # create directory for this run
            if (run_eval):
                if not(globcount is None):
                    os.makedirs((path)+str(globcount))
            else:
                input("Copy files now,..")

            result = np.zeros((np.shape(x)[0],1))
            # for very parameter run the evaluation,...
            for i in range(np.shape(x)[0]):
                path_curr_params = (path) + str(globcount) + '/' + str(i) + '/'
                if (run_eval):
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
                    result[i,:] = run_param_rollout(globcount, str(path+'pos.pkl'), path_curr_params, str(path_curr_params+'params.pkl'), run_eval)
                else:
                    result[i,:] = run_param_rollout_real(globcount, str(path+'pos.pkl'), path_curr_params, str(path_curr_params+'params.pkl'), run_eval)

            return result

    def __call__(self, x, path, globcount=None, run_eval=True):
        #noise_std = 0.
        return self.function(x,path,globcount,run_eval) #+ noise_std * np.random.randn(*x.shape)

    def x_sample(self, n=1):
        return np.random.uniform(self.x_min, self.x_max, (n, self.d_x)).reshape((n, self.d_x))

    def x_domain(self, n=1):
        return np.linspace(self.x_min, self.x_max, n).reshape((n, self.d_x))





EXPERIMENTS = {
    "toy"       : Toy,
    "rrc_v1"    : RRC_v1,
}

MODELS = {
    "gp":  GaussianProcess,
}

def main(args, res_dir):

    print ("HAAALLLOOO")

    modify_and_push_json()

    number_rollouts_per_param = NUM_ROLLOUTS_PER_SAMPLE
    # Sample potential goal and target positions.
    # WRITE DIFFICULTY LEVEL TO FILE
    with open(('./content/diff.txt'), 'w') as f:
        f.write(str(DIFFICULTY_LEVEL))

    define_sample_fct(SAMPLE_FCT, number_rollouts_per_param, ("./content"+'/'), path2=(res_dir+'/'))
    push_github("push_position_file")
    #input ("WAIT")
    logging_txt_file = (res_dir+'/' + "RESULTS_LOGGING.txt")

    reference_path = (res_dir+'/')

    logger.info("Args:")
    for k, v in vars(args).items():
        logger.info("%s: %s", k, v)

    task = EXPERIMENTS[args.experiment]()


    if False:
        if task.plot_model:
            x = task.x_domain(1000)
            X = to_tensor(x)
            y = task(x)
            Y = to_tensor(y)

    y_opt = task.y_opt

    y_history = np.zeros((args.n_seeds, task.num_iter+1))
    y_opt_est = np.zeros((args.n_seeds, task.num_iter))
    bounds = torch.stack([to_tensor(task.x_min), to_tensor(task.x_max)]).to(TORCH_DEVICE)

    # TODO: find out what this means -> I think not needed
    if (False):
        # constuct 'cubature points' from bounds for whitening
        pts = np.vstack([task.x_min, task.x_max])
        x_vert = np.hstack([m.reshape((-1, 1))
                            for m in np.meshgrid(*(pts[:, i] for i in range(task.d_x)))])
        y_vert = task(x_vert, reference_path)
        x_bounds = to_tensor(x_vert)
        y_bounds = to_tensor(y_vert)
    globcount_thres = -1
    for s in range(args.n_seeds):
        with open(logging_txt_file,"w") as f:
            f.write("SEED: " + str(s) + '\n')
        globcount = 0

        model = MODELS[args.model](task.d_x, task.param_normalizer, **vars(args))

        # Generate initial data
        _X_train_raw = task.x_sample(task.num_init_samples)
        _pt = task.x_sample(1)
        _X_train_raw = np.concatenate([_X_train_raw] + [_pt])

        if (globcount<=globcount_thres):
            print (_X_train_raw)
            _y_train_raw = task(_X_train_raw, reference_path, globcount,run_eval=False)
        else:
            _y_train_raw = task(_X_train_raw, reference_path, globcount, run_eval=True)
        globcount += 1

        X_train, _y_train = map(to_tensor, [_X_train_raw, _y_train_raw])
        X_train = X_train.to(TORCH_DEVICE)
        _y_train = _y_train.to(TORCH_DEVICE)
        y_train = _y_train

        # Get best observed value from dataset
        with open(logging_txt_file,"a") as f:
            f.write("iteration: " + str(globcount) + '\n')
            f.write("max val: " + str(y_train.max().item()) + '\n')
            f.write("params:  " + str(X_train[y_train.argmax().item(),:].tolist()) + '\n')

        y_history[s, 0]  = y_train.max().item()

        logger.info("seed iter best opt | y_chosen | y_optimal")

        # Bayesian Optimization Loop
        for i in tqdm(range(task.num_iter), total=task.num_iter):
            _model = model.fit(X_train, y_train)

            # # FROM FABIO:
            # # Acquisition functions
            # if self.acq_fcn_type == "UCB":
            #     acq_fcn = UpperConfidenceBound(gp, beta=self.acq_param.get("beta", 0.1), maximize=True)
            # elif self.acq_fcn_type == "EI":
            #     acq_fcn = ExpectedImprovement(gp, best_f=cands_values_stdized.max().item(), maximize=True)
            # elif self.acq_fcn_type == "PI":
            #     acq_fcn = ProbabilityOfImprovement(gp, best_f=cands_values_stdized.max().item(), maximize=True)
            # else:
            #     raise pyrado.ValueErr(given=self.acq_fcn_type, eq_constraint="'UCB', 'EI', 'PI'")
            #
            # # Optimize acquisition function and get new candidate point
            # cand_norm, _ = optimize_acqf(
            #     acq_function=acq_fcn,
            #     bounds=to.stack([to.zeros(self.ddp_space.flat_dim), to.ones(self.ddp_space.flat_dim)]).to(
            #         dtype=to.float32
            #     ),
            #     q=1,
            #     num_restarts=self.acq_restarts,
            #     raw_samples=self.acq_samples,
            # )

            acq = ExpectedImprovement(_model, best_f=model.data_normalizer.standardize_wo_calculation(y_train).max().item(), maximize=True)

            # Optimize acquisition function
            # q represents addidional points to sample
            # SEE DOCS: Expected imrpovement only supports q=1
            candidate, acq_value = optimize_acqf(
                acq_function=acq,
                bounds=torch.stack([torch.zeros(np.shape(task.param_normalizer.bound_lo)[0]), torch.ones(np.shape(task.param_normalizer.bound_lo)[0])]).to(dtype=torch.float32).to(TORCH_DEVICE),
                q=1,
                num_restarts=task.num_acq_restarts,
                raw_samples=task.num_acq_samples
            )
            candidate = model.param_normalizer.project_back(candidate)


            with torch.no_grad():
                x_new = candidate
                if (globcount <= globcount_thres):
                    print(x_new.cpu().numpy())
                    y_new = to_tensor(task(x_new.cpu().numpy(), reference_path, globcount,run_eval=False)).to(TORCH_DEVICE)
                else:
                    y_new = to_tensor(task(x_new.cpu().numpy(), reference_path, globcount, run_eval=True)).to(
                        TORCH_DEVICE)
                globcount += 1
                y_est_m, y_est_v = model.predict(x_new.view(1, -1).to(TORCH_DEVICE))
                y_opt_m, y_opt_v = model.predict(to_tensor(task.x_opt).view(1, -1).to(TORCH_DEVICE))
                y_opt_est[s, i] = y_opt_m.cpu().numpy()

                # Update dataset
                X_train = torch.cat([X_train, x_new])
                _y_train = torch.cat([_y_train, y_new])
                y_train = _y_train

                # Update best observed value list
                y_history[s, i+1] = y_train.max().item()
                logger.info(f"{s:2} {i+1:3} {y_history[s, i+1]:.4f} {y_opt:.4f} | {np.asscalar(y_est_m.cpu().numpy()):.4f}+/-{np.asscalar(torch.sqrt(y_est_v).cpu().numpy()):.4f} | {np.asscalar(y_opt_m.cpu().numpy()):.4f}+/-{np.asscalar(torch.sqrt(y_opt_v).cpu().numpy()):.4f}")
                with open(logging_txt_file, "a") as f:
                    f.write("iteration: " + str(globcount) + '\n')
                    f.write("max val: " + str(y_train.max().item()) + '\n')
                    f.write("params:  " + str(X_train[y_train.argmax().item(), :].tolist()) + '\n')


                if task.plot_model and args.plot:
                    utility = acq(X.unsqueeze(1)) # why does it need batch x q x d??
                    plot_model(_model, X, Y, utility, X_train, _y_train, x_new, y_new, acq_value, y_history, y_opt, f"{s}_{i}")

                if args.plot and args.experiment != "toy":
                    f, ax = plt.subplots(task.d_x)
                    ax = [ax] if task.d_x == 1 else ax
                    for d, a in enumerate(ax):
                        x0 = np.copy(task.x_opt)
                        xN = np.copy(task.x_opt)
                        x0[d] = 0.
                        xN[d] = 1.
                        _x = np.linspace(x0, xN, 1000)
                        x = _x[:, d]
                        _y_m, _y_v = model.predict(to_tensor(_x).view(1000, -1).to(TORCH_DEVICE))
                        _y_m = _y_m.cpu()
                        _y_v = _y_m.cpu()
                        _y = task(_x, reference_path)
                        _ytrain = task(X_train.cpu().numpy(), reference_path)
                        a.plot(x, _y_m, 'b')
                        plot_uncertainty(a, x, _y_m, _y_v, stds=[3, 2, 1])
                        a.plot(x, _y, 'r')
                        a.plot(X_train.cpu().numpy()[:, d], _ytrain, 'm*')
                    plt.savefig(os.path.join(res_dir, f"slice_{i}.png"),
                        bbox_inches='tight', format='png')
                    plt.close(f)

    save_metrics(y_history, y_opt, y_opt_est, res_dir)




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
    parser.add_argument("experiment", help="Task to run", choices=EXPERIMENTS.keys())
    parser.add_argument("model", help="Model to use", choices=MODELS.keys())
    parser.add_argument("--path", help="Path where results are to be stored", default="")

    parser.add_argument("-n", "--name", help="Optional name for experiment", default="")
    parser.add_argument("--lr", type=float, help="Learning rate", default=1e-4)
    parser.add_argument("-e", "--epochs", type=int, help="Epochs", default=100)
    parser.add_argument("-s", "--seed", type=int, help="Random seed", default=0)
    parser.add_argument("-k", "--n-seeds", type=int, help="Number of seeds", default=1)
    parser.add_argument('-d', '--hidden-dims', default=[50, 50], nargs='+', type=int, help="Specify dimensions of hidden layers. (e.g. 50 50)")
    parser.add_argument('-a', '--activation', default='tanh', help="Specify nonlinear activation function. (e.g. 'relu', 'tanh', ...)")
    parser.add_argument('--use-kld', default=False, action='store_true', help="Enable latent derivative objective?")
    parser.add_argument("--prior-variance", type=float, help="for KL", default=1.)
    parser.add_argument('--release', default=False, action='store_true', help="'Proper experiment' - not debug, no timestamp on experiment dir")
    parser.add_argument('--plot', default=False, action='store_true', help="Do plotting")
    parser.add_argument('--cpu', default=False, action='store_true', help="Force CPU")
    args = parser.parse_args()

    if args.cpu:
        TORCH_DEVICE = torch.device("cpu")

    desc = f"{args.model}{args.activation}_{'' if args.use_kld else 'No'}LD" if args.model != "gp" else args.model
    name = f"BO_{args.experiment}_{desc}_{args.seed}_{args.name if not args.release else ''}"
    res_dir = make_results_folder(name, datetime=not args.release, abs_path = args.path)

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
