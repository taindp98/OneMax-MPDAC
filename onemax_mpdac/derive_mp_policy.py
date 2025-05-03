import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from onemax_mpdac.eval import ollga_mp_single_run
import torch
import os

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--type", type=str, default="lbd1")
    parser.add_argument("--is-discrete", action="store_true", default=False)
    args = parser.parse_args()
    n = args.n
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
    portfolio = [1, 2, 4, 8, 16, 32, 64]

    theory_policy = [np.sqrt(n / (n - i)) for i in range(n)]
    theory_policy = [
        int(np.ceil(v)) if v - np.floor(v) > 0.5 else int(np.floor(v))
        for v in theory_policy
    ]

    portfolio = [k for k in sorted(portfolio, reverse=True) if k < n]
    if args.is_discrete:
        theory_policy = np.array(
            [min(portfolio, key=lambda x: abs(x - v)) for v in theory_policy]
        )
    else:
        theory_policy = np.array(theory_policy)

    if args.type == "lbd1_alpha_lbd2":
        lbd1 = []
        for sol in range(n):
            if sol / n <= 0.95:
                lbd1.append(1)
            else:
                lbd1.append(theory_policy[sol])
        lbd2 = 2 * theory_policy
        alpha = []
        for sol in range(n):
            if sol / n <= 0.95:
                alpha.append(1e-3)
            else:
                alpha.append(1)
        alpha = np.array(alpha)
        beta = np.ones(n)
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
    elif args.type == "lbd1":
        lbd1 = []
        for sol in range(n):
            if sol / n <= 0.95:
                lbd1.append(1)
            else:
                lbd1.append(theory_policy[sol])
        lbd2 = lbd1
        alpha = np.ones(n)
        beta = np.ones(n)
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
    elif args.type == "alpha":
        lbd1 = theory_policy
        lbd2 = lbd1
        alpha = []
        for sol in range(n):
            if sol / n <= 0.95:
                alpha.append(1e-3)
            else:
                alpha.append(1)
        alpha = np.array(alpha)
        beta = np.ones(n)
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
    elif args.type == "lbd2":
        lbd1 = theory_policy
        lbd2 = 2 * theory_policy
        alpha = np.ones(n)
        beta = np.ones(n)
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
    elif args.type == "lbd1_alpha":
        lbd1 = []
        for sol in range(n):
            if sol / n <= 0.95:
                lbd1.append(1)
            else:
                lbd1.append(theory_policy[sol])
        lbd2 = lbd1
        alpha = []
        for sol in range(n):
            if sol / n <= 0.95:
                alpha.append(1e-3)
            else:
                alpha.append(1)
        alpha = np.array(alpha)
        beta = np.ones(n)
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
    elif args.type == "lbd1_lbd2":
        lbd1 = []
        for sol in range(n):
            if sol / n <= 0.95:
                lbd1.append(1)
            else:
                lbd1.append(theory_policy[sol])
        lbd2 = 2 * theory_policy
        alpha = np.ones(n)
        beta = np.ones(n)
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
    elif args.type == "alpha_lbd2":
        lbd1 = theory_policy
        lbd2 = 2 * theory_policy
        alpha = []
        for sol in range(n):
            if sol / n <= 0.95:
                alpha.append(1e-3)
            else:
                alpha.append(1)
        alpha = np.array(alpha)
        beta = np.ones(n)
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
    else:
        raise ValueError("Invalid type of policy")

    runtimes = Parallel(n_jobs=20)(
        delayed(ollga_mp_single_run)(bench_params, eval_env_params, policy, i)
        for i in tqdm(
            range(1000),
            desc=f"Running problem: n={n}",
            disable=False,
            ncols=100,
        )
    )
    outputs = {
        "policy": policy,
        "runtimes": runtimes,
    }
    mean = np.mean(runtimes)
    std = np.std(runtimes)
    print(f"n = {n}: {mean:.3f} ± {std:.2f}")
    if args.is_discrete:
        savedir = "outputs/discrete_derived_policies"
    else:
        savedir = "outputs/continuous_derived_policies"
    os.makedirs(savedir, exist_ok=True)
    torch.save(outputs, f"{savedir}/n{n}#{args.type}.pth")
