import os
import physo
import torch
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import physo.learn.monitoring as monitoring

# Parallel config :
# Parallel mode may cause issues due to the number of samples, non-parallel mode is recommended
# Single core with so many samples will actually use up to 10 cores via pytorch parallelization along sample dim
PARALLEL_MODE_DEFAULT = False
N_CPUS_DEFAULT = 1

#
# SEED = 0
# FRAC_REALS = 1.0
# N_SPE_FREE_PARAMS   = 2
# N_CLASS_FREE_PARAMS = 1
# BARY = "hydro" # "dmo" or "hydro"
# # Parallel config
# PARALLEL_MODE = False
# N_CPUS        = -1


# Example run
# python nihao_run.py --nspe 2 --nclass 1 --bary hydro --trial 0 --frac_real 1.0

# ---------------------------------------------------- SCRIPT ARGS -----------------------------------------------------
parser = argparse.ArgumentParser(description="Runs a nihao run.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-s", "--nspe", default=2, help="Number of SPE free params.")
parser.add_argument("-c", "--nclass", default=1, help="Number of CLASS free params.")
parser.add_argument("-b", "--bary", default="hydro", help="Baryonic or DMO: 'hydro' or 'dmo'.")
parser.add_argument("-t", "--trial", default=0, help="Trial number (sets seed accordingly).")
parser.add_argument("-f", "--frac_real", default=1., help="Fraction of realizations to use (rounded up). Use eg. 1e-6 which will be rounded up to use only one realization.")
parser.add_argument("-bs", "--batch_size", default=2000, help="Batch size.")
parser.add_argument("-p", "--parallel_mode", default=PARALLEL_MODE_DEFAULT, help="Should parallel mode be used.")
parser.add_argument("-ncpus", "--ncpus", default=N_CPUS_DEFAULT, help="Nb. of CPUs to use")
config = vars(parser.parse_args())

# Nb of spe free params
N_SPE_FREE_PARAMS = int(config["nspe"])
assert N_SPE_FREE_PARAMS > 0, "N_SPE_FREE_PARAMS must be > 0"

# Nb of class free params
N_CLASS_FREE_PARAMS = int(config["nclass"])

# Baryonic or DMO
BARY = str(config["bary"])

# Trial number
SEED = int(config["trial"])

# Fraction of realizations to use
FRAC_REALS = float(config["frac_real"])

# Batch size
BATCH_SIZE = int(config["batch_size"])

# Parallel config
PARALLEL_MODE = bool(config["parallel_mode"])
N_CPUS = int(config["ncpus"])


# ---------------------------------------------------- SCRIPT ARGS -----------------------------------------------------


# PATHS
# Defining source data abs path before changing directory
# At top level, in the original project dir
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

DATA_PATH = os.path.join(PROJECT_ROOT, "NIHAO_data/%s_profiles/" % BARY)
METADATA_PATH = os.path.join(DATA_PATH, "MvirRvir.dat")

RUN_NAME = "NIHAO_runb_%s_%i_%f_nspe%i_nclass%i_bs%i" % (BARY, SEED, FRAC_REALS, N_SPE_FREE_PARAMS, N_CLASS_FREE_PARAMS, BATCH_SIZE)

RUN_DIR = os.path.join(PROJECT_ROOT, RUN_NAME)

if not os.path.exists(RUN_DIR):
    os.makedirs(RUN_DIR)

os.chdir(RUN_DIR)


# Simulations' metadata
metadata = pd.read_csv(METADATA_PATH, sep=' ', header=None, names=['sim', 'rvir', 'mvir', 'rvir_bn', 'mvir_bn'])
metadata = metadata.astype({'sim': str, 'rvir': float, 'mvir': float, 'rvir_bn': float, 'mvir_bn': float})

# Removing profiles with missing rvir_bn or mvir_bn
metadata = metadata[~(metadata["rvir_bn"].isna() | metadata["mvir_bn"].isna())]

# Selecting rows with metadata["sim"] starting with 'g1'
# metadata = metadata[metadata["sim"].str.startswith('g1')]

# Subsampling data
metadata = metadata.sample(frac=FRAC_REALS, random_state=SEED)
n_reals = len(metadata)

# Saving subsampled metadata
metadata.to_csv('subsample.csv', index=False)

# Histograms of rvir_bn and mvir_bn
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
fig.suptitle('%i profiles' % n_reals)
ax[0].hist(metadata["rvir_bn"], bins=30, histtype='step', color='k')
ax[0].set_xlabel(r'$R_{vir}\ [kpc]$')
ax[1].hist(metadata["mvir_bn"], bins=30, histtype='step', color='k')
ax[1].set_xlabel(r'$M_{vir}\ [M_{\odot}]$')
fig.savefig('subsample_histo.png', dpi=300)
plt.show()

# Selection names
sim_names = metadata["sim"]


# Getting profiles
def get_profile(sim_name):
    df = pd.read_csv(DATA_PATH + '%s_profile.dat' % (sim_name), sep=' ', header=None, names=['r', 'n', 'rho'])
    df = df.astype({'r': float, 'n': int, 'rho': float})
    # Selecting profiles points with more than 1000 particles
    df = df[df["n"] > 1000]
    # Metadata for the profile
    md = metadata[metadata["sim"] == sim_name]
    rvir_bn = md["rvir_bn"].values[0]
    mvir_bn = md["mvir_bn"].values[0]
    # Normalizing profile
    df["r"] = df["r"] / rvir_bn
    df["rho"] = df["rho"] / (mvir_bn / ((4 / 3) * np.pi * (rvir_bn**3)))
    return df


multi_X = []
multi_y = []
multi_y_weights = []
for sim_name in sim_names:
    df = get_profile(sim_name)
    r = df["r"].values
    rho = df["rho"].values
    n = df["n"].values
    # Poisson uncertainty from n : y_unc = np.sqrt(n) / n = 1 / np.sqrt(n)
    y_weight = np.sqrt(n)      # (n_samples,)
    X = np.stack([r], axis=0)  # (1, n_samples)
    y = rho                    # (n_samples,)
    # Appending to multi_X, multi_y, multi_y_weights
    multi_X.append(X)
    multi_y.append(y)
    multi_y_weights.append(y_weight)
n_samples_per_dataset = np.array([X.shape[1] for X in multi_X])

# Let's evaluate/ fit quality on log(y) vs log(f(x))
multi_y = [np.log(y) for y in multi_y]

# Constants
FIXED_CONSTS = [1.,]
CLASS_FREE_CONSTS_NAMES = ["c%i" % (i) for i in range(N_CLASS_FREE_PARAMS)]
CLASS_FREE_CONSTS_UNITS = [[0, 0] for i in range(N_CLASS_FREE_PARAMS)]

SPE_FREE_CONSTS_NAMES = ["rho0",] + ["rs%i" % (i) for i in range(N_SPE_FREE_PARAMS - 1)]
SPE_FREE_CONSTS_UNITS = [[1, -3],] + [[0, 1] for _ in range(N_SPE_FREE_PARAMS - 1)]

Y_UNITS = [1, -3]
X_UNITS = [[0, 1]]

free_const_opti_args = {'loss': "MSE", 'method': 'LBFGS', 'method_args': {'n_steps': 50, 'tol': 1e-99, 'lbfgs_func_args': {'max_iter': 4, 'line_search_fn': "strong_wolfe", }, }}

# CONFIG
OP_NAMES = ["add", "sub", "mul", "div", "inv", "n2", "sqrt", "neg", "log", "exp"]
RUN_CONFIG = physo.config.config2b.config2b


# k0 / ((r/k1) * (1 + r/k1)**2)
# ENFORCING EQUATION TO START WITH LOG
# target_prog_str = ["div", "rho0", "mul", "div", "r", "rs0", "n2", "add", "1.0", "div", "r", "rs0",]
# cheater_prior_config = ('SymbolicPrior', {'expression': target_prog_str})
# RUN_CONFIG["priors_config"].append(cheater_prior_config)
# tmp

# candidate_wrapper = None
def candidate_wrapper(func, X):
    return torch.log(torch.abs(func(X)))


# ENFORCING NEW FREE CONSTS PARAMS
RUN_CONFIG["free_const_opti_args"] = free_const_opti_args

# Hack here
RUN_CONFIG["learning_config"]["batch_size"] = 100  # 20 # tmp

MAX_N_EVALUATIONS = 50000 + 1  # int(2.5 * 1e5) + 1
# Allowed to search in an infinitely large search space, research will be stopped by MAX_N_EVALUATIONS
N_EPOCHS = int(1e99)


class DummyVisualiser:
    def visualise(self, *args, **kwargs):
        return


run_logger = lambda: monitoring.RunLogger(save_path='SR.log', do_save=True)
run_visualiser = lambda: DummyVisualiser()
# run_visualiser = lambda: monitoring.RunVisualiser(epoch_refresh_rate=1, save_path='SR_curves.png', do_show=False, do_prints=True, do_save=True)


# Running SR task

expression, logs = physo.ClassSR(multi_X, multi_y, multi_y_weights=multi_y_weights, X_names=["r",], X_units=X_UNITS, y_name="rho", y_units=Y_UNITS,
                                 # Fixed constants
                                 fixed_consts=FIXED_CONSTS,
                                 # Free constants names (for display purposes)
                                 class_free_consts_names=CLASS_FREE_CONSTS_NAMES, class_free_consts_units=CLASS_FREE_CONSTS_UNITS,
                                 # Spe free constants names (for display purposes)
                                 spe_free_consts_names=SPE_FREE_CONSTS_NAMES, spe_free_consts_units=SPE_FREE_CONSTS_UNITS,
                                 # Operations allowed
                                 op_names=OP_NAMES,
                                 # Wrapper
                                 candidate_wrapper=candidate_wrapper,
                                 # Run config
                                 run_config=RUN_CONFIG,
                                 # Run monitoring
                                 get_run_logger=run_logger, get_run_visualiser=run_visualiser,
                                 # Stopping condition
                                 stop_reward=1.1,  # not stopping even if perfect 1.0 reward is reached
                                 max_n_evaluations=MAX_N_EVALUATIONS, epochs=N_EPOCHS,
                                 # Parallel mode
                                 parallel_mode=PARALLEL_MODE, n_cpus=N_CPUS)


# for i_real in range(n_reals):
#     # Getting prediction
#     X = multi_X[i_real]
#     y = multi_y[i_real]
#     y_weight = multi_y_weights[i_real]
#     y_pred = expression.execute(torch.tensor(X), i_realization=i_real).cpu().detach().numpy()
#     # Plotting
#     fig, ax = plt.subplots(1, 1, figsize=(10, 5))
#     ax.errorbar(X[0], y, yerr=1 / np.sqrt(y_weight), fmt='.', markersize=1., label='Data')
#     ax.plot(X[0], y_pred, label='Fit')
#     ax.set_xlabel('r')
#     ax.set_ylabel('log(rho)')
#     ax.legend()
#     fig.savefig('fit_%i.png' % (i_real), dpi=300)

# Plotting results
for i_real in range(n_reals):
    X = multi_X[i_real]
    y = multi_y[i_real]
    y_weight = multi_y_weights[i_real]

    X_torch = torch.tensor(X)
    y_pred_t = expression.execute(X_torch, i_realization=i_real)
    y_pred = y_pred_t.cpu().detach().numpy().reshape(-1)

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.errorbar(X[0], y, yerr=1 / np.sqrt(y_weight), fmt='.', markersize=1., label='Data')
    ax.plot(X[0], y_pred, label='Fit')
    ax.set_xlabel('r')
    ax.set_ylabel('log(rho)')
    ax.legend()
    fig.savefig(f'fit_{i_real}.png', dpi=300)
    plt.close(fig)


print(None)
