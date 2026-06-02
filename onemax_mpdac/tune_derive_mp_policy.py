import numpy as np
from onemax_mpdac.eval import ollga_multi_param
import os
import datetime
import time
import json
import argparse
from smac import Scenario, MultiFidelityFacade
from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter
from smac.intensifier.hyperband_utils import get_n_trials_for_hyperband_multifidelity
import multiprocessing

PROBLEM_SIZES = [100, 500]

def get_lbd1(breaking_point: float, n: int, theory_policy: np.ndarray):
    lbd1 = []
    for sol in range(n):
        if sol / n <= breaking_point:
            lbd1.append(1)
        else:
            lbd1.append(theory_policy[sol])
    return lbd1


def get_lbd2(amplified_factor: float, theory_policy: np.ndarray):
    lbd2 = amplified_factor * theory_policy
    return lbd2


def get_alpha(breaking_point: float, n: int, abyss: float):
    alpha = []
    for sol in range(n):
        if sol / n <= breaking_point:
            alpha.append(abyss)
        else:
            alpha.append(1)
    alpha = np.array(alpha)
    return np.array(alpha)


def get_beta(n: int):
    beta = np.ones(n)
    return beta


def objective_function(cfg, seed=None, budget=100, instance=None):
    """
    cfg: A ConfigSpace Configuration, can be accessed like a dict
    """
    lbd1_breaking_point = float(cfg["lbd1_breaking_point"])
    lbd2_amplified_factor = float(cfg["lbd2_amplified_factor"])
    alpha_breaking_point = float(cfg["alpha_breaking_point"])
    alpha_abyss = float(cfg["alpha_abyss"])
    norm_runtimes = []
    for n in args.problem_sizes:
        bench_params = {
            "name": "OLLGAFactTheory",
            "discrete_action": True,
            "action_choices": [
                [1, 2, 4, 8, 16, 32, 64],
                [0.25, 0.542, 0.833, 1.125, 1.417, 1.708, 2.0],
                [1, 2, 4, 8, 16, 32, 64],
                [0.25, 0.542, 0.833, 1.125, 1.417, 1.708, 2.0],
            ],
            "problem": "OneMax",
            "instance_set_path": f"om_ollga_{n}_medium.csv",
            "observation_description": "n,f(x)",
            "reward_choice": "imp_minus_evals_shifting",
            "alias": "evenly_spread",
            "seed": 123,
        }
        eval_env_params = {"reward_choice": "minus_evals", "cutoff": 100000.0}
        theory_policy = [np.sqrt(n / (n - i)) for i in range(n)]
        theory_policy = [
            int(np.ceil(v)) if v - np.floor(v) > 0.5 else int(np.floor(v))
            for v in theory_policy
        ]
        theory_policy = np.array(theory_policy)

        lbd1 = get_lbd1(
            breaking_point=lbd1_breaking_point, n=n, theory_policy=theory_policy
        )
        lbd2 = get_lbd2(
            amplified_factor=lbd2_amplified_factor, theory_policy=theory_policy
        )
        alpha = get_alpha(breaking_point=alpha_breaking_point, n=n, abyss=alpha_abyss)
        beta = get_beta(n=n)

        policy = []
        for i in range(n):
            policy.append(
                [
                    np.int64(lbd1[i]),
                    np.float64(alpha[i]),
                    np.int64(lbd2[i]),
                    np.float64(beta[i]),
                ]
            )

        budget = int(budget)
        runtimes = []
        for i in range(budget):
            runtime = ollga_multi_param(bench_params, eval_env_params, policy, i)
            runtimes.append(runtime)
        # norm_runtimes = np.mean(runtimes) / n
        norm_runtimes.append(np.mean(runtimes) / n)
    overall_norm_runtime = np.mean(norm_runtimes)
    # SMAC always minimizes the objective
    return overall_norm_runtime


if __name__ == "__main__":
    ##
    argparser = argparse.ArgumentParser(description="SMAC3 Hyperparameter Optimization")
    argparser.add_argument(
        "--total-budget",
        type=int,
        default=1e6,
        help="Total budget for the optimization in terms of fidelity units (default: 1e6)",
    )
    argparser.add_argument(
        "--total-cpus",
        type=int,
        default=multiprocessing.cpu_count(),
        help="Total number of CPUs available for the optimization (default: all available CPUs)",
    )
    argparser.add_argument(
        "--problem-sizes",
        type=int,
        nargs="+",
        default=PROBLEM_SIZES,
        help="List of problem sizes to optimize over (default: 100, 200, 500, 1000, 2000, 5000, 10000)",
    )
    args = argparser.parse_args()
    # 1. Define Search Space
    cs = ConfigurationSpace()
    cs.add_hyperparameters(
        [
            UniformFloatHyperparameter(
                "lbd1_breaking_point", lower=0.5, upper=1 - 1e-5
            ),
            UniformFloatHyperparameter("lbd2_amplified_factor", lower=1.0, upper=5.0),
            UniformFloatHyperparameter(
                "alpha_breaking_point", lower=0.5, upper=1 - 1e-5
            ),
            UniformFloatHyperparameter("alpha_abyss", lower=1e-3, upper=0.1),
        ]
    )
    # n_workers = max(1, args.total_cpus // args.cpus_per_trial // 2)
    n_workers = max(1, args.total_cpus)  # Use all available CPUs
    # 2. Define Scenario
    ## https://automl.github.io/SMAC3/development/examples/2_multi_fidelity/3_specify_HB_via_total_budget.html#sphx-glr-examples-2-multi-fidelity-3-specify-hb-via-total-budget-py
    n_trials = get_n_trials_for_hyperband_multifidelity(
        total_budget=args.total_budget,  # this is the total optimization budget we specify in terms of fidelity units
        min_budget=100,  # This influences the Hyperband rounds, minimum budget per trial
        max_budget=1000,  # This influences the Hyperband rounds, maximum budget per trial
        eta=3,  # This influences the Hyperband rounds
        print_summary=True,
    )

    scenario = Scenario(
        configspace=cs,
        n_trials=n_trials,
        output_directory="outputs/smac3",
        n_workers=n_workers,
        deterministic=True,
        min_budget=100,  # e.g., start with 10 repetitions
        max_budget=1000,  # e.g., up to 1000 repetitions (your old default)
    )

    # 3. Run SMAC Optimization Loop
    start_time = time.time()

    smac = MultiFidelityFacade(
        scenario,
        objective_function,  # We pass the target function here
        overwrite=True,  # Overrides any previous results that are found that are inconsistent with the meta-data
        intensifier=MultiFidelityFacade.get_intensifier(scenario=scenario, eta=3),
    )
    
    incumbent = smac.optimize()
    end_time = time.time()
    elapsed_time = end_time - start_time

    # 4. Save Best Config and Elapsed Time
    best_config = incumbent.get_dictionary()
    # Get the cost/objective value for this configuration from SMAC's run history
    runhistory = smac.runhistory
    best_cost = runhistory.get_cost(incumbent)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join("outputs", "smac3")
    os.makedirs(save_dir, exist_ok=True)
    data = {
        "best_config": best_config,
        "best_cost": best_cost,
        "elapsed_time": elapsed_time,
    }
    with open(os.path.join(save_dir, f"{timestamp}.json"), "w") as f:
        json.dump(data, f, indent=4)

    # Get cost of default configuration
    default_cost = smac.validate(cs.get_default_configuration())
    print(f"Default cost: {default_cost}")

    # Let's calculate the cost of the incumbent
    incumbent_cost = smac.validate(incumbent)
    print(f"Incumbent cost: {incumbent_cost}")
