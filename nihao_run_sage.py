import os
import torch
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import time
# PhySO imports
import physo
import physo.learn.monitoring as monitoring
import physo.physym.batch_execute as bexec
# Local imports
from functions_sage import check_analytical_properties


def check_profile(expr_str: str, params: list[str]) -> dict:
    """
    expr_str : PhySO expression string, e.g. '((rs0+-(r))*...)'
    params   : list of parameter names in expr_str, e.g. ['rho0', 'rs1', 'rs0']

    Returns a dict: { 'Density': True/False, 'Enclosed Mass': ..., ... }.
    """
    df = check_analytical_properties(expr_str, free_consts_names=params)
    if df.empty:
        return {}
    return {row["Property"]: bool(row["symb_condition"]) for _, row in df.iterrows()}


PARALLEL_MODE_DEFAULT = False
N_CPUS_DEFAULT = 1

# Example run
# python nihao_run.py --nspe 2 --nclass 1 --bary hydro --trial 0 --frac_real 1.0

# ---------------------------------------------------- SCRIPT ARGS -----------------------------------------------------
parser = argparse.ArgumentParser(description="Runs a nihao run.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-s", "--nspe", default=2, help="Number of SPE free params.")
parser.add_argument("-c", "--nclass", default=1, help="Number of CLASS free params.")
parser.add_argument("-b", "--bary", default="hydro", help="Baryonic or DMO: 'hydro' or 'dmo'.")
parser.add_argument("-t", "--trial", default=0, help="Trial number (sets seed accordingly).")
parser.add_argument("-f", "--frac_real", default=0.1, help="Fraction of realizations to use (rounded up). Use eg. 1e-6 which will be rounded up to use only one realization.")
parser.add_argument("-bs", "--batch_size", default=2000, help="Batch size.")
parser.add_argument("-p", "--parallel_mode", default=PARALLEL_MODE_DEFAULT, help="Should parallel mode be used.")
parser.add_argument("-ncpus", "--ncpus", default=N_CPUS_DEFAULT, help="Nb. of CPUs to use")
config = vars(parser.parse_args())

N_SPE_FREE_PARAMS = int(config["nspe"])
assert N_SPE_FREE_PARAMS > 0, "N_SPE_FREE_PARAMS must be > 0"
N_CLASS_FREE_PARAMS = int(config["nclass"])
BARY = str(config["bary"])  # Baryonic or DMO
SEED = int(config["trial"])  # Trial number
FRAC_REALS = float(config["frac_real"])  # Fraction of realizations to use
BATCH_SIZE = int(config["batch_size"])
PARALLEL_MODE = bool(config["parallel_mode"])
N_CPUS = int(config["ncpus"])

# ---------------------------------------------------- SCRIPT ARGS -----------------------------------------------------

if __name__ == "__main__":
    # PATHS
    # Defining source data abs path before changing directory
    # At top level, in the original project dir
    PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

    DATA_PATH = os.path.join(PROJECT_ROOT, "NIHAO_data/%s_profiles/" % BARY)
    METADATA_PATH = os.path.join(DATA_PATH, "MvirRvir.dat")

    RUN_NAME = "NIHAO_runb_%s_%i_%i_nspe%i_nclass%i_bs%i" % (BARY, SEED, FRAC_REALS, N_SPE_FREE_PARAMS, N_CLASS_FREE_PARAMS, BATCH_SIZE)

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

    # CUSTOM REWARD : including analytical properties analysis

    # During programs evaluation, should parallel execution be used ?
    USE_PARALLEL_EXE = False  # Only worth it if n_all_samples > 1e6
    USE_PARALLEL_OPTI_CONST = False  # Only worth it if batch_size > 1k

    def my_SquashedNRMSE(y_target, y_pred, y_weights=1.):
        """
        Squashed NRMSE reward.
        Parameters
        ----------
        y_target : torch.tensor of shape (?,) of float
            Target output data.
        y_pred   : torch.tensor of shape (?,) of float
            Predicted data.
        y_weights : torch.tensor of shape (?,) of float, optional
            Weights for each data point. By default, no weights are used.
        Returns
        -------
        reward : torch.tensor float
            Reward encoding prediction vs target discrepancy in [0,1].
        """
        sigma_targ = y_target.std()
        # Computing error with weights
        err = y_weights * (y_target - y_pred)**2  # (?,)
        RMSE = torch.sqrt(torch.mean(err))
        NRMSE = (1 / sigma_targ) * RMSE
        reward = 1 / (1 + NRMSE)
        return reward

    reward_config = {"reward_function": my_SquashedNRMSE, "zero_out_unphysical": True, "zero_out_duplicates": False, "keep_lowest_complexity_duplicate": False,
        # "parallel_mode" : True,
        # "n_cpus"        : None,
    }

    def my_RewardsComputer(programs,
                           X,
                           y_target,
                           n_samples_per_dataset,
                           y_weights=1.,
                           free_const_opti_args=None,
                           reward_function=my_SquashedNRMSE,
                           zero_out_unphysical=False,
                           zero_out_duplicates=False,
                           keep_lowest_complexity_duplicate=False,
                           parallel_mode=False,
                           n_cpus=None,
                           progress_bar=False,
                           ):
        """
        Computes rewards of programs on X data accordingly with target y_target and reward reward_function using torch
        for acceleration.
        Parameters
        ----------
        programs : Program.VectProgram
            Programs contained in batch to evaluate.
        X : torch.tensor of shape (n_dim, ?,) of float
            Values of the input variables of the problem with n_dim = nb of input variables.
        y_target : torch.tensor of shape (?,) of float
            Values of the target symbolic function on input variables contained in X_target.
        n_samples_per_dataset : array_like of shape (n_realizations,) of int
            We assume that X contains multiple datasets with samples of each ataset following each other and each portion
            of X corresponding to a dataset should be treated with its corresponding dataset specific free constants values.
            n_samples_per_dataset is the number of samples for each dataset. Eg. [90, 100, 110] for 3 datasets, this will
            assume that the first 90 samples of X are for the first dataset, the next 100 for the second and the last 110
            for the third.
        y_weights : torch.tensor of shape (?,) of float, optional
            Weights for each data point.
        free_const_opti_args : dict or None, optional
            Arguments to pass to free_const.optimize_free_const for free constant optimization. By default,
            free_const.DEFAULT_OPTI_ARGS arguments are used.

        reward_function : callable
            Function that taking y_target (torch.tensor of shape (?,) of float), y_pred (torch.tensor of shape (?,)
            of float) and  optionally  y_weights (torch.tensor of shape (?,) of float, optional) as key arguments and
            returning a float reward of an individual program.
        zero_out_unphysical : bool
            Should unphysical programs be zeroed out ?
        zero_out_duplicates : bool
            Should duplicate programs (equal symbolic value when simplified) be zeroed out ?
        keep_lowest_complexity_duplicate : bool
            If True, when eliminating duplicates (via zero_out_duplicates = True), the least complex duplicate is kept, else
            a random duplicate is kept.
        Returns
        -------
        rewards : numpy.array of shape (?,) of float
            Rewards of programs.
        """

        # ----- SETUP -----

        # mask : should program reward NOT be zeroed out ie. is program invalid ?
        # By default all programs are considered valid
        mask_valid = np.full(shape=programs.batch_size, fill_value=True, dtype=bool)                         # (batch_size,)

        # ----- PHYSICALITY -----
        if zero_out_unphysical:
            # mask : is program physical
            mask_is_physical = programs.is_physical                                                          # (batch_size,)
            # Update mask to zero out unphysical programs
            mask_valid = (mask_valid & mask_is_physical)                                                     # (batch_size,)

        # ----- DUPLICATES -----
        if zero_out_duplicates:
            # Compute rewards (even if programs have non-optimized free consts) to serve as a unique numeric identifier of
            # functional forms (programs having equivalent forms will have the same reward).

            # Only use parallel mode if enabled in function param and in USE_PARALLEL_EXE flag.
            # This way users can use flags to specifically enable or disable parallel exe and/or const opti.
            parallel_mode_exe = parallel_mode and USE_PARALLEL_EXE
            rewards_non_opt = programs.batch_exe_reward(X=X,
                                                        y_target=y_target,
                                                        y_weights=y_weights,
                                                        reward_function=reward_function,
                                                        n_samples_per_dataset=n_samples_per_dataset,
                                                        mask=mask_valid,
                                                        pad_with=0.0,
                                                        # Parallel related
                                                        parallel_mode=parallel_mode_exe,
                                                        n_cpus=n_cpus,
                                                        )
            # mask : is program a unique one we should keep ?
            # By default, all programs are eliminated.
            mask_unique_keep = np.full(shape=programs.batch_size, fill_value=False, dtype=bool)              # (batch_size,)
            # Identifying unique programs.
            unique_rewards, unique_idx = np.unique(rewards_non_opt, return_index=True)                       # (n_unique,), (n_unique,)
            if keep_lowest_complexity_duplicate:
                unique_idx_lowest_comp = []
                # Iterating through unique rewards
                for r in unique_rewards:
                    # mask: does program have current unique reward ?
                    mask_have_r = (rewards_non_opt == r)                                                     # (batch_size,)
                    # complexities of programs having current unique reward
                    complexities_at_r = programs.n_complexity[mask_have_r]                                   # (n_at_r,)
                    # idx in batch of program having current unique reward of the lowest complexity
                    idx_lowest_comp = np.arange(programs.batch_size)[mask_have_r][complexities_at_r.argmin()]
                    unique_idx_lowest_comp.append(idx_lowest_comp)
                # Idx of unique programs (having the lowest complexity among their duplicates)
                unique_idx_lowest_comp = np.array(unique_idx_lowest_comp)
                # Keeping the lowest complexity duplicate of unique programs
                mask_unique_keep[unique_idx_lowest_comp] = True
            else:
                # Keeping first occurrences of unique programs (random)
                mask_unique_keep[unique_idx] = True                                                          # (n_unique,)
            # Update mask to zero out duplicate programs
            mask_valid = (mask_valid & mask_unique_keep)                                                     # (batch_size,)

        # ----- ANALYTIC PROPERTIES CHECKING -----
        zero_out_property_violating = True
        t0 = time.perf_counter()
        mask_violates = np.full(shape=programs.batch_size, fill_value=False, dtype=bool)              # (batch_size,)
        if zero_out_property_violating:
            for i_expr in range(programs.batch_size):
                if mask_valid[i_expr]:  # Only check properties of programs that are still valid at this stage (not unphysical or duplicates)
                    print("Checking properties of expr %i / %i" % (i_expr + 1, programs.batch_size))
                    expression = programs[i_expr]
                    expr_str = expression.get_infix_str()
                    free_const_names = programs.library.free_const_names
                    flags = check_profile(expr_str=expr_str, params=free_const_names)
                    print("flags", flags)
                    # Violates if any of the following are False :
                    try:
                        #properties = np.array([flags['Enclosed Mass'], flags['Circular Velocity'], flags['Potential'], flags['Radial Velocity Dispersion']])
                        properties = np.array([flags['Enclosed Mass'], flags['Circular Velocity'], flags['Potential'],])
                        property_violating = np.any(properties == False)
                    except KeyError:
                        # If any of the properties is not in the flags dict, we consider that the program violates the properties (to be conservative)
                        property_violating = True
                else:
                    property_violating = False
                mask_violates[i_expr] = property_violating
            # Update mask to zero out property violating programs
            mask_valid = (mask_valid & ~mask_violates)                                                     # (batch_size,)
        t1 = time.perf_counter()
        print("Analytic properties checking time: %.2f seconds for %i programs" % (t1 - t0, programs.batch_size))

        # ----- FREE CONST OPTIMIZATION -----
        # If there are free constants in the library, we have to optimize them
        if programs.library.n_free_const > 0:
            # Only use parallel mode if enabled in function param and in USE_PARALLEL_OPTI_CONST flag.
            # This way users can use flags to specifically enable or disable parallel exe and/or const opti.
            parallel_mode_const_opti = parallel_mode and USE_PARALLEL_OPTI_CONST
            # Opti const
            # batch_optimize_free_const (programs, X, y_target, args_opti = free_const_opti_args, mask_valid = mask_valid)
            programs.batch_optimize_constants(X=X,
                                              y_target=y_target,
                                              free_const_opti_args=free_const_opti_args,
                                              y_weights=y_weights,
                                              mask=mask_valid,
                                              n_samples_per_dataset=n_samples_per_dataset,
                                              # Parallel related
                                              parallel_mode=parallel_mode_const_opti,
                                              n_cpus=n_cpus)

        # ----- REWARDS -----
        # If rewards were already computed at the duplicate elimination step and there are no free constants in the library
        # No need to recompute rewards.
        if zero_out_duplicates and programs.library.n_free_const == 0:
            rewards = rewards_non_opt
        # Else we need to compute rewards
        else:
            # Only use parallel mode if enabled in function param and in USE_PARALLEL_EXE flag.
            # This way users can use flags to specifically enable or disable parallel exe and/or const opti.
            parallel_mode_exe = parallel_mode and USE_PARALLEL_EXE
            rewards = programs.batch_exe_reward(X=X,
                                                y_target=y_target,
                                                y_weights=y_weights,
                                                reward_function=reward_function,
                                                n_samples_per_dataset=n_samples_per_dataset,
                                                mask=mask_valid,
                                                pad_with=0.0,
                                                # Parallel related
                                                parallel_mode=parallel_mode_exe,
                                                n_cpus=n_cpus,
                                                )

        # Applying mask (this is redundant)
        rewards = rewards * mask_valid.astype(float)
        # Safety to avoid nan rewards (messes up gradients)
        rewards = np.nan_to_num(rewards, nan=0.)

        return rewards

    def my_make_RewardsComputer(reward_function=my_SquashedNRMSE,
                                zero_out_unphysical=False,
                                zero_out_duplicates=False,
                                keep_lowest_complexity_duplicate=False,
                                # Parallel related
                                parallel_mode=True,
                                n_cpus=None,
                                ):
        """
        Helper function to make custom reward computing function.
        Parameters
        ----------
        reward_function : callable
            Function that taking y_target (torch.tensor of shape (?,) of float), y_pred (torch.tensor of shape (?,)
            of float) and  optionally  y_weights (torch.tensor of shape (?,) of float, optional) as key arguments and
            returning a float reward of an individual program.
        zero_out_unphysical : bool
            Should unphysical programs be zeroed out ?
        zero_out_duplicates : bool
            Should duplicate programs (equal symbolic value when simplified) be zeroed out ?
        keep_lowest_complexity_duplicate : bool
            If True, when eliminating duplicates (via zero_out_duplicates = True), the least complex duplicate is kept, else
            a random duplicate is kept.
        parallel_mode : bool
            Tries to use parallel execution if True (availability will be checked by batch_execute.ParallelExeAvailability),
            execution in a loop else.
        n_cpus : int or None
            Number of CPUs to use when running in parallel mode. By default, uses the maximum number of CPUs available.
        Returns
        -------
        rewards_computer : callable
             Custom reward computing function taking programs (vect_programs.VectPrograms), X (torch.tensor of shape (n_dim,?,)
             of float), y_target (torch.tensor of shape (?,) of float), y_weights (torch.tensor of shape (?,) of float),
             n_samples_per_dataset (array_like of shape (n_realizations,) of int) and free_const_opti_args as key arguments
             and returning reward for each program (array_like of float).
        """
        # Check that parallel execution is available on this system
        recommended_config = bexec.ParallelExeAvailability()
        is_parallel_mode_available_on_system = recommended_config["parallel_mode"]
        # If not available and parallel_mode was still instructed warn and disable
        if not is_parallel_mode_available_on_system and parallel_mode:
            bexec.ParallelExeAvailability(verbose=True) # prints explanation
            warnings.warn("Parallel mode is not available on this system, switching to non parallel mode.")
            parallel_mode = False

        # rewards_computer
        def rewards_computer(programs, X, y_target, y_weights, n_samples_per_dataset, free_const_opti_args):
            R = my_RewardsComputer(programs=programs, X=X, y_target=y_target, y_weights=y_weights, n_samples_per_dataset=n_samples_per_dataset, free_const_opti_args=free_const_opti_args,
                                   # Frozen args
                                   reward_function=reward_function, zero_out_unphysical=zero_out_unphysical, zero_out_duplicates=zero_out_duplicates, keep_lowest_complexity_duplicate=keep_lowest_complexity_duplicate,
                                   # Parallel related
                                   parallel_mode=parallel_mode, n_cpus=n_cpus, 
                                   )
            return R

        return rewards_computer

    RUN_CONFIG["learning_config"]["rewards_computer"] = my_make_RewardsComputer(**reward_config)
    RUN_CONFIG["learning_config"]["custom_rewards_computer"] = True

    def candidate_wrapper(func, X):
        return torch.log(torch.abs(func(X)))

    # ENFORCING NEW FREE CONSTS PARAMS
    RUN_CONFIG["free_const_opti_args"] = free_const_opti_args

    # Hack here
    RUN_CONFIG["learning_config"]["batch_size"] = BATCH_SIZE  # 20 # tmp
    RUN_CONFIG["learning_config"]["risk_factor"] = 0.1  # Increasing risk factor to allow for more exploration
    MAX_N_EVALUATIONS = 50000 + 1  # int(2.5 * 1e5) + 1
    N_EPOCHS = int(1e99)  # Allowed to search in an infinitely large search space, research will be stopped by MAX_N_EVALUATIONS

    class DummyVisualiser:
        def visualise(self, *args, **kwargs):
            return
    # run_visualiser = lambda: DummyVisualiser()

    run_logger = lambda: monitoring.RunLogger(save_path='SR.log', do_save=True)
    run_visualiser = lambda: monitoring.RunVisualiser(epoch_refresh_rate=1, save_path='SR_curves.png', do_show=False, do_prints=True, do_save=True)

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

    print(None)
