import sys
import yaml
import os
import torch
import numpy as np
import random
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns


from dacbench.benchmarks import (
    RLSTheoryBenchmark,
    OLLGATheoryBenchmark,
    OLLGAFactTheoryBenchmark,
    OLLGACombTheoryBenchmark,
)

from dacbench.benchmarks import (
    OLLGAFactL1L2TheoryBenchmark,
    OLLGAFactL1L2MTheoryBenchmark,
    OLLGAFactL1L2CTheoryBenchmark,
    OLLGAFactL1TheoryBenchmark,
    OLLGAFactL1MTheoryBenchmark,
    OLLGAFactL1CTheoryBenchmark,
    OLLGAFactL1MCTheoryBenchmark,
)

default_exp_params = {
    "n_cores": 1,
    "n_episodes": 1e6,
    "n_steps": 1e6,
    "eval_interval": 2000,
    "n_eval_episodes_per_instance": 50,
    "save_agent_at_every_eval": False,
    "seed": 123,
    "eval_mode": "formula",
    "use_cuda": False,
    "log_level": 1,
}

default_bench_params = {
    "name": "Theory",
    "alias": "evenly_spread",
    "discrete_action": True,
    "action_choices": [1, 17, 33],
    "problem": "LeadingOnes",
    "instance_set_path": "lo_rls_50_random.csv",
    "observation_description": "n,f(x)",
    "reward_choice": "imp_minus_evals",
    "seed": 123,
}

default_eval_env_params = {
    "reward_choice": "minus_evals",
    "cutoff": 1e5,
}


def read_config(config_yml_fn: str = "output/config.yml"):
    with open(config_yml_fn, "r") as f:
        params = yaml.safe_load(f)

    for key in default_exp_params:
        if key not in params["experiment"]:
            params["experiment"][key] = default_exp_params[key]

    for key in default_bench_params:
        if key not in params["bench"]:
            params["bench"][key] = default_bench_params[key]

    train_env_params = eval_env_params = None
    if "train_env" in params:
        train_env_params = params["train_env"]
    if "eval_env" in params:
        eval_env_params = params["eval_env"]
        for key in default_eval_env_params:
            if key not in eval_env_params:
                eval_env_params[key] = default_eval_env_params[key]
    return (
        params["experiment"],
        params["bench"],
        params["agent"],
        train_env_params,
        eval_env_params,
    )


def make_env(bench_params, env_config=None, test_env=False):
    """
    env_config will override bench_params
    """
    bench_class = globals()[bench_params["name"] + "Benchmark"]

    params = bench_params.copy()
    del params["name"]
    if env_config:
        for name, val in env_config.items():
            params[name] = val

    # pprint(params)
    bench = bench_class(config=params)
    env = bench.get_environment(test_env)
    # env = wrappers.FlattenObservation(env)
    return env


def object_to_dict(obj):
    if isinstance(obj, dict):
        return {k: object_to_dict(v) for k, v in obj.items()}
    elif hasattr(obj, "__dict__"):
        return {k: object_to_dict(v) for k, v in obj.__dict__.items()}
    elif isinstance(obj, list):
        return [object_to_dict(i) for i in obj]
    else:
        return obj


class Config:
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                value = Config(value)
            setattr(self, key, value)


def load_config(config_path):
    with open(config_path, "r") as file:
        config_dict = yaml.safe_load(file)
    return Config(config_dict)


def seed_everything(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_time_str():
    """
    Get the current time as a string
    """
    now = datetime.now()
    date_str = now.strftime("%Y%m%d")
    time_str = now.strftime("%H%M%S")
    return date_str, time_str


def object_to_dict(obj):
    if isinstance(obj, dict):
        return {k: object_to_dict(v) for k, v in obj.items()}
    elif hasattr(obj, "__dict__"):
        return {k: object_to_dict(v) for k, v in obj.__dict__.items()}
    elif isinstance(obj, list):
        return [object_to_dict(i) for i in obj]
    else:
        return obj


def get_theory_policy(n: int = 2000):
    lbd1 = [np.sqrt(n / (n - i)) for i in range(n)]
    lbd1 = [
        int(np.ceil(v)) if v - np.floor(v) > 0.5 else int(np.floor(v)) for v in lbd1
    ]
    lbd2 = lbd1
    alpha = np.ones(n)
    beta = np.ones(n)
    policy = []
    for i in range(n):
        policy.append(
            [
                np.int64(lbd1[i]),
                np.float64(alpha[i]),
                np.clip(lbd1[i] * alpha[i] / n, 0, 1),
                np.int64(lbd2[i]),
                np.float64(beta[i]),
                np.clip(beta[i] / lbd2[i], 0, 1),
            ]
        )
    policy = np.array(policy)
    return policy


def plot_policies(results_fpath: str):
    eval_data = np.load(results_fpath, allow_pickle=True)
    instance_set = eval_data["instance_set"].item()
    i = 0
    inst_id = eval_data["inst_ids"][i]
    instance = instance_set[inst_id]
    n = instance["size"]

    best_idx = np.argmin(eval_data["eval_runtime_means"])
    eval_policies = eval_data[
        "eval_policies"
    ]  # Assuming this is now a dictionary of policies
    policy = np.array(eval_policies[best_idx][0])
    p = np.array(policy[:, 0] * policy[:, 1] / n)
    c = np.array(policy[:, 3] / policy[:, 2])
    ## clip c to [0, 1]
    c = np.clip(c, 0, 1)

    rl_policy = np.array([policy[:, 0], policy[:, 1], p, policy[:, 2], policy[:, 3], c])

    rl_policy = np.stack(rl_policy, axis=1)
    policies = {
        "rl": rl_policy,
        "theory": get_theory_policy(n),
    }
    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    sns.set(style="whitegrid")
    # Flatten the axes array for easier iteration
    fontsize = 14
    axes = axes.flatten()

    # Plot each column in a separate subplot
    for i in range(6):
        for idx, (policy_name, policy_data) in enumerate(policies.items()):
            # line_style = line_styles[idx % len(line_styles)]
            # marker = "o" if policy_name == "RL" else "t"
            x_data = np.linspace(0, 1, policy_data.shape[0])
            if policy_name == "theory":
                label = r"$\pi_{THEORY}$"
                axes[i].plot(
                    x_data,
                    policy_data[:, i],
                    label=label,
                    linewidth=3,
                    linestyle="-",
                    alpha=0.6,
                    color="lightseagreen",
                )
            else:
                ## set line style
                label = r"$\pi_{RL}$"
                axes[i].plot(
                    x_data,
                    policy_data[:, i],
                    label=label,
                    linewidth=2,
                    linestyle="--",
                    alpha=0.8,
                    color="crimson",
                )
        axes[i].grid(True, linestyle="--", alpha=0.5)
        ## set x limit from n//2 to n
        axes[i].set_xlim(0.5, 1.01)
        if i == 0:
            axes[i].set_ylim(0, 34)

    # Set titles for each subplot
    axes[0].set_ylabel(r"$\lambda_m$", fontsize=fontsize, rotation=True, labelpad=15)
    axes[0].set_xlabel(r"$f(x)/n$")
    axes[1].set_ylabel(r"$\alpha$", fontsize=fontsize, rotation=True, labelpad=15)
    axes[1].set_xlabel(r"$f(x)/n$")
    axes[2].set_ylabel(r"$p$", fontsize=fontsize, rotation=True, labelpad=15)
    axes[2].set_xlabel(r"$f(x)/n$")
    axes[3].set_ylabel(r"$\lambda_c$", fontsize=fontsize, rotation=True, labelpad=15)
    axes[3].set_xlabel(r"$f(x)/n$")
    axes[4].set_ylabel(r"$\beta$", fontsize=fontsize, rotation=True, labelpad=15)
    axes[4].set_xlabel(r"$f(x)/n$")
    axes[5].set_ylabel(r"$c$", fontsize=fontsize, rotation=True, labelpad=15)
    axes[5].set_xlabel(r"$f(x)/n$")
    # Add a legend to the last subplot
    # Collect unique handles and labels from the first subplot
    handles, labels = axes[0].get_legend_handles_labels()
    ## set plot title
    # Add a single legend at the bottom of the figure
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.05),
        ncol=len(labels),
        fontsize=fontsize,
    )
    # Adjust layout and show the plot
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(
        os.path.join(os.path.dirname(results_fpath), "Policy Comparison.png"),
        dpi=300,
        bbox_inches="tight",
    )
