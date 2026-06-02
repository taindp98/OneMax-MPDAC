import argparse
import os
from typing import Optional
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from onemax_mpdac.utils import make_env, read_config, get_time_str, NormalizeActionWrapper
import shutil
import numpy as np
import yaml
from joblib import Parallel, delayed
from tqdm import tqdm
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from onemax_mpdac.eval import ollga_multi_param, ollga_single_param

class MultiParamEval:
    def __init__(
        self,
        agent,
        env,
        eval_env,
        bench_params,
        eval_env_params,
        n_eval_episodes_per_instance: int = 100,
        log_path: Optional[str] = None,
        save_agent_at_every_eval: bool = False,
        verbose: int = 1,
        name: str = "",
        n_cpus: int = 1,
        env_norm: bool = False,
    ):
        self.env = env
        self.eval_env = eval_env
        self.verbose = verbose
        self.agent = agent
        self.name = name
        self.log_path = log_path
        self.n_cpus = n_cpus
        if save_agent_at_every_eval:
            assert (
                log_path is not None
            ), "ERROR: log_path must be specified when save_agent_at_every_eval=True"

        # create log_path folder if it doesn't exist
        if self.log_path is not None:
            os.makedirs(self.log_path, exist_ok=True)

        # Detailed logs will be written in <log_path>/evaluations.npz
        self.detailed_log_path = None
        if log_path is not None:
            self.detailed_log_path = os.path.join(self.log_path, "evaluations")

        self.n_eval_episodes_per_instance = n_eval_episodes_per_instance
        self.save_agent_at_every_eval = save_agent_at_every_eval

        # we will calculate optimal policy and its runtime for each instance
        self.instance_set = eval_env.instance_set

        # list of inst_id (keys of self.instance_set)
        self.inst_ids = eval_env.instance_id_list

        # best/last mean_runtime of each instance
        self.best_mean_runtime = [np.inf] * len(self.inst_ids)
        self.last_mean_runtime = [np.inf] * len(self.inst_ids)

        # element i^th: optimal policy for instance self.inst_ids[i]
        self.optimal_policies = []
        self.optimal_runtime_means = []
        self.optimal_runtime_stds = []

        # evaluation timesteps
        self.eval_timesteps = []

        # element i^th:
        #   - policy at self.eval_timesteps[i]
        #   - its runtime per instance (sorted by self.inst_ids)
        #   - a list of number of decisions made per episode for each instance (for TempoRL)
        self.eval_policies = []
        self.eval_policies_unclipped = []
        self.eval_runtime_means = []
        self.eval_runtime_stds = []
        self.eval_n_decisions = []

        if hasattr(eval_env, "action_choices"):
            self.action_choices = eval_env.action_choices
            self.discrete_portfolio = True
        else:
            self.discrete_portfolio = False

        # if self.verbose >= 1:
        #     print("Optimal policies:")
        self.bench_params = bench_params
        self.eval_env_params = eval_env_params
        for inst_id in self.inst_ids:
            inst = self.instance_set[inst_id]
            n = inst["size"]

            # get the optimal policy
            policy = [np.sqrt(n / (n - i)) for i in range(n)]
            policy = [
                int(np.ceil(v)) if v - np.floor(v) > 0.5 else int(np.floor(v))
                for v in policy
            ]
            if self.discrete_portfolio:
                portfolio = [
                    k
                    for k in sorted(eval_env.action_choices[inst_id][0], reverse=True)
                    if k < n
                ]
                ## map the optimal policy to the nearest element in discrete portfolio
                policy = [min(portfolio, key=lambda x: abs(x - v)) for v in policy]

            self.optimal_policies.append(policy)
            # calculate the runtime of the optimal policy
            runtimes = Parallel(n_jobs=self.n_cpus)(
                delayed(ollga_single_param)(
                    self.bench_params, self.eval_env_params, policy, i
                )
                for i in range(self.n_eval_episodes_per_instance)
            )
            runtime_mean = np.mean(runtimes)
            runtime_std = np.std(runtimes)
            self.optimal_runtime_means.append(runtime_mean)
            self.optimal_runtime_stds.append(runtime_std)

        self.env_norm = env_norm

    def _normalize_obs(self, obs, obs_mean, obs_var, epsilon=1e-8):
        """
        Manually normalize observations to match the VecNormalize training statistics.
        """
        return (obs - obs_mean) / np.sqrt(obs_var + epsilon)

    def eval(self, n_steps) -> bool:
        self.eval_timesteps.append(n_steps)

        policies = []
        runtime_means = []
        runtime_stds = []

        for inst_id in self.inst_ids:
            inst = self.instance_set[inst_id]
            n = inst["size"]
            # policy_unclipped = self.agent.get_actions_for_all_states(n)
            obs = [[n, i] for i in range(n)]
            if self.env_norm:
                obs = self._normalize_obs(
                    np.array(obs),
                    self.env.obs_rms.mean,
                    self.env.obs_rms.var,
                )
                print(
                    f"Normalized obs using obs_rms: mean {self.env.obs_rms.mean}, var {self.env.obs_rms.var}"
                )
            actions, _ = self.agent.policy.predict(obs, deterministic=True)
            policy = []
            if self.discrete_portfolio:
                for fitness, sel in enumerate(actions):
                    # lbd1_idx, mr_idx, lbd2_idx, cr_idx = sel
                    # lambda1 = self.action_choices[inst_id][0][lbd1_idx]
                    # mutation_rate = self.action_choices[inst_id][1][mr_idx]
                    # lambda2 = self.action_choices[inst_id][2][lbd2_idx]
                    # crossover_rate = self.action_choices[inst_id][3][cr_idx]
                    if self.bench_params["name"] == "OLLGATheoryPPO":
                        lbd1_idx, mr_idx, lbd2_idx, cr_idx = sel
                        lambda1 = self.action_choices[inst_id][0][lbd1_idx]
                        mutation_rate = self.action_choices[inst_id][1][mr_idx]
                        lambda2 = self.action_choices[inst_id][2][lbd2_idx]
                        crossover_rate = self.action_choices[inst_id][3][cr_idx]
                    elif self.bench_params["name"] == "OLLGAL1L2TheoryPPO":
                        lbd1_idx, lbd2_idx = sel
                        lambda1 = self.action_choices[inst_id][0][lbd1_idx]
                        mutation_rate = 1
                        lambda2 = self.action_choices[inst_id][1][lbd2_idx]
                        crossover_rate = 1
                    elif self.bench_params["name"] == "OLLGAL1L2MTheoryPPO":
                        lbd1_idx, mutation_idx, lbd2_idx = sel
                        lambda1 = self.action_choices[inst_id][0][lbd1_idx]
                        mutation_rate = self.action_choices[inst_id][1][mutation_idx]
                        lambda2 = self.action_choices[inst_id][2][lbd2_idx]
                        crossover_rate = 1
                    elif self.bench_params["name"] == "OLLGAL1L2CTheoryPPO":
                        lbd1_idx, lbd2_idx, cr_idx = sel
                        lambda1 = self.action_choices[inst_id][0][lbd1_idx]
                        mutation_rate = 1
                        lambda2 = self.action_choices[inst_id][1][lbd2_idx]
                        crossover_rate = self.action_choices[inst_id][2][cr_idx]
                    elif self.bench_params["name"] == "OLLGAL1TheoryPPO":
                        lambda1 = self.action_choices[inst_id][0][sel[0]]
                        mutation_rate = 1
                        lambda2 = lambda1
                        crossover_rate = 1
                    elif self.bench_params["name"] == "OLLGAL1MTheoryPPO":
                        lbd1_idx, mutation_idx = sel
                        lambda1 = self.action_choices[inst_id][0][lbd1_idx]
                        mutation_rate = self.action_choices[inst_id][1][mutation_idx]
                        lambda2 = lambda1
                        crossover_rate = 1
                    elif self.bench_params["name"] == "OLLGAL1CTheoryPPO":
                        lbd1_idx, crossover_idx = sel
                        lambda1 = self.action_choices[inst_id][0][lbd1_idx]
                        crossover_rate = self.action_choices[inst_id][1][crossover_idx]
                        lambda2 = lambda1
                        mutation_rate = 1
                    elif self.bench_params["name"] == "OLLGAL1MCTheoryPPO":
                        lbd1_idx, mutation_idx, crossover_idx = sel
                        lambda1 = self.action_choices[inst_id][0][lbd1_idx]
                        mutation_rate = self.action_choices[inst_id][1][mutation_idx]
                        crossover_rate = self.action_choices[inst_id][2][crossover_idx]
                        lambda2 = lambda1
                    else:
                        raise NotImplementedError

                    policy.append(
                        [
                            np.int64(lambda1),
                            np.float64(mutation_rate),
                            np.int64(lambda2),
                            np.float64(crossover_rate),
                        ]
                    )
            else:
                ## Continuous action space
                if "rescale_action" in self.bench_params:
                    if "independent_action" in self.bench_params:
                        # env_normalizer = HybridNormalizeWrapper(self.eval_env, independent=self.bench_params["independent_action"])
                        env_normalizer = NormalizeActionWrapper(
                            self.eval_env,
                            independent=self.bench_params["independent_action"],
                        )
                    else:
                        # env_normalizer = HybridNormalizeWrapper(self.eval_env)
                        env_normalizer = NormalizeActionWrapper(self.eval_env)
                    policy = []
                    for fitness, raw_action in enumerate(actions):
                        scaled_action = env_normalizer.action(raw_action)
                        policy.append(scaled_action)
                else:
                    policy = actions

            policies.append(policy)

            # calculate runtime of current policy
            # set self.eval_env's instance_set to a single instance (inst_id)
            self.eval_env.instance_id_list = [inst_id]
            self.eval_env.instance_index = 0
            self.eval_env.instance_set = {inst_id: inst}
            runtimes = Parallel(n_jobs=self.n_cpus)(
                delayed(ollga_multi_param)(
                    self.bench_params, self.eval_env_params, policy, i
                )
                for i in tqdm(
                    range(self.n_eval_episodes_per_instance),
                    desc=f"Eval Problem n: {n} @ step: {n_steps}",
                    disable=False,
                    ncols=100,
                )
            )

            runtime_mean = np.mean(runtimes)
            runtime_std = np.std(runtimes)

            self.eval_env.instance_id_list = self.inst_ids
            self.eval_env.instance_set = self.instance_set

            runtime_means.append(runtime_mean)
            runtime_stds.append(runtime_std)
            # print(f"Ldb1: {np.array(policy)[:,0]}")
            print(f"Runtime: {runtime_mean} +/- {runtime_std}")

        if self.detailed_log_path is not None:
            # save eval statistics
            self.eval_policies.append(policies)
            self.eval_runtime_means.append(runtime_means)
            self.eval_runtime_stds.append(runtime_stds)

            np.savez(
                self.detailed_log_path,
                inst_ids=self.inst_ids,
                optimal_policies=np.array(self.optimal_policies, dtype=object),
                optimal_runtime_means=self.optimal_runtime_means,
                optimal_runtime_stds=self.optimal_runtime_stds,
                eval_timesteps=self.eval_timesteps,
                eval_policies=np.array(self.eval_policies, dtype=object),
                eval_runtime_means=self.eval_runtime_means,
                eval_runtime_stds=self.eval_runtime_stds,
                instance_set=self.instance_set,
            )
            # save current model
            if self.save_agent_at_every_eval:
                self.agent.save(os.path.join(self.log_path, f"model_{n_steps}"))

        # update best_mean_runtime
        if self.log_path:
            self.last_mean_runtime = runtime_means
            # how many instances where we get infs
            n_best_infs = sum([v == np.inf for v in self.best_mean_runtime])
            n_cur_infs = sum([v == np.inf for v in runtime_means])
            # mean runtime across instances, inf excluded
            best_overall_mean = np.ma.masked_invalid(self.best_mean_runtime).mean()
            cur_overall_mean = np.ma.masked_invalid(runtime_means).mean()
            # update best
            if (n_cur_infs < n_best_infs) or (
                (n_cur_infs == n_best_infs) and (cur_overall_mean < best_overall_mean)
            ):
                self.best_mean_runtime = runtime_means
                
                if self.log_path:
                    self.agent.save(os.path.join(self.log_path, "best_model"))

        return runtime_means, runtime_stds


class SingleParamEval:
    def __init__(
        self,
        agent,
        eval_env,
        bench_params,
        eval_env_params,
        n_eval_episodes_per_instance: int = 100,
        log_path: Optional[str] = None,
        save_agent_at_every_eval: bool = False,
        verbose: int = 1,
        name: str = "",
        n_cpus: int = 1,
    ):
        self.eval_env = eval_env
        self.verbose = verbose
        self.agent = agent
        self.name = name
        self.log_path = log_path
        self.n_cpus = n_cpus
        if save_agent_at_every_eval:
            assert (
                log_path is not None
            ), "ERROR: log_path must be specified when save_agent_at_every_eval=True"

        # create log_path folder if it doesn't exist
        if self.log_path is not None:
            os.makedirs(self.log_path, exist_ok=True)

        # Detailed logs will be written in <log_path>/evaluations.npz
        self.detailed_log_path = None
        if log_path is not None:
            self.detailed_log_path = os.path.join(self.log_path, "evaluations")

        self.n_eval_episodes_per_instance = n_eval_episodes_per_instance
        self.save_agent_at_every_eval = save_agent_at_every_eval

        # we will calculate optimal policy and its runtime for each instance
        self.instance_set = eval_env.instance_set

        # list of inst_id (keys of self.instance_set)
        self.inst_ids = eval_env.instance_id_list

        # best/last mean_runtime of each instance
        self.best_mean_runtime = [np.inf] * len(self.inst_ids)
        self.last_mean_runtime = [np.inf] * len(self.inst_ids)

        # element i^th: optimal policy for instance self.inst_ids[i]
        self.optimal_policies = []
        self.optimal_runtime_means = []
        self.optimal_runtime_stds = []

        # evaluation timesteps
        self.eval_timesteps = []

        # element i^th:
        #   - policy at self.eval_timesteps[i]
        #   - its runtime per instance (sorted by self.inst_ids)
        #   - a list of number of decisions made per episode for each instance (for TempoRL)
        self.eval_policies = []
        self.eval_policies_unclipped = []
        self.eval_runtime_means = []
        self.eval_runtime_stds = []
        self.eval_n_decisions = []

        if hasattr(eval_env, "action_choices"):
            self.action_choices = eval_env.action_choices
            self.discrete_portfolio = True
        else:
            self.discrete_portfolio = False

        # if self.verbose >= 1:
        #     print("Optimal policies:")
        self.bench_params = bench_params
        self.eval_env_params = eval_env_params
        for inst_id in self.inst_ids:
            inst = self.instance_set[inst_id]
            n = inst["size"]

            # get the optimal policy
            policy = [np.sqrt(n / (n - i)) for i in range(n)]
            policy = [
                int(np.ceil(v)) if v - np.floor(v) > 0.5 else int(np.floor(v))
                for v in policy
            ]
            if self.discrete_portfolio:
                portfolio = eval_env.action_choices[inst_id]
                ## map the optimal policy to the nearest element in discrete portfolio
                policy = [min(portfolio, key=lambda x: abs(x - v)) for v in policy]

            self.optimal_policies.append(policy)
            # calculate the runtime of the optimal policy
            runtimes = Parallel(n_jobs=self.n_cpus)(
                delayed(ollga_single_param)(
                    self.bench_params, self.eval_env_params, policy, i
                )
                for i in range(self.n_eval_episodes_per_instance)
            )
            runtime_mean = np.mean(runtimes)
            runtime_std = np.std(runtimes)

            self.optimal_runtime_means.append(runtime_mean)
            self.optimal_runtime_stds.append(runtime_std)

    def eval(self, n_steps) -> bool:
        self.eval_timesteps.append(n_steps)

        policies = []
        runtime_means = []
        runtime_stds = []

        for inst_id in self.inst_ids:
            inst = self.instance_set[inst_id]
            n = inst["size"]
            # policy_unclipped = self.agent.get_actions_for_all_states(n)
            obs = [[n, i] for i in range(n)]
            actions, _ = self.agent.policy.predict(obs, deterministic=True)

            if self.discrete_portfolio:
                policy = []
                for fitness, sel in enumerate(actions):
                    lambda1 = self.action_choices[inst_id][sel]
                    policy.append(lambda1)
            else:
                raise NotImplementedError(
                    "Action choices for continuous portfolio are not implemented yet."
                )

            policies.append(policy)

            # calculate runtime of current policy
            # set self.eval_env's instance_set to a single instance (inst_id)
            self.eval_env.instance_id_list = [inst_id]
            self.eval_env.instance_index = 0
            self.eval_env.instance_set = {inst_id: inst}
            runtimes = Parallel(n_jobs=self.n_cpus)(
                delayed(ollga_single_param)(
                    self.bench_params, self.eval_env_params, policy, i
                )
                for i in tqdm(
                    range(self.n_eval_episodes_per_instance),
                    desc=f"Eval Problem n: {n} @ step: {n_steps}",
                    disable=False,
                    ncols=100,
                )
            )

            runtime_mean = np.mean(runtimes)
            runtime_std = np.std(runtimes)

            self.eval_env.instance_id_list = self.inst_ids
            self.eval_env.instance_set = self.instance_set

            runtime_means.append(runtime_mean)
            runtime_stds.append(runtime_std)

        if self.detailed_log_path is not None:
            # save eval statistics
            self.eval_policies.append(policies)
            self.eval_runtime_means.append(runtime_means)
            self.eval_runtime_stds.append(runtime_stds)

            np.savez(
                self.detailed_log_path,
                inst_ids=self.inst_ids,
                optimal_policies=np.array(self.optimal_policies, dtype=object),
                optimal_runtime_means=self.optimal_runtime_means,
                optimal_runtime_stds=self.optimal_runtime_stds,
                eval_timesteps=self.eval_timesteps,
                eval_policies=np.array(self.eval_policies, dtype=object),
                eval_runtime_means=self.eval_runtime_means,
                eval_runtime_stds=self.eval_runtime_stds,
                instance_set=self.instance_set,
            )
            # save current model
            if self.save_agent_at_every_eval:
                self.agent.save(os.path.join(self.log_path, f"model_{n_steps}"))

        # update best_mean_runtime
        if self.log_path:
            self.last_mean_runtime = runtime_means
            # how many instances where we get infs
            n_best_infs = sum([v == np.inf for v in self.best_mean_runtime])
            n_cur_infs = sum([v == np.inf for v in runtime_means])
            # mean runtime across instances, inf excluded
            best_overall_mean = np.ma.masked_invalid(self.best_mean_runtime).mean()
            cur_overall_mean = np.ma.masked_invalid(runtime_means).mean()
            # update best
            if (n_cur_infs < n_best_infs) or (
                (n_cur_infs == n_best_infs) and (cur_overall_mean < best_overall_mean)
            ):
                self.best_mean_runtime = runtime_means
                # if self.verbose >= 1:
                #     print(
                #         f"\t[env: {self.name}] New best mean runtime! ({runtime_means})"
                #     )
                if self.log_path:
                    self.agent.save(os.path.join(self.log_path, "best_model"))

        return runtime_means, runtime_stds


class EvalCallback(BaseCallback):
    """
    A custom callback that evaluates the agent every `eval_interval` steps.
    """

    def __init__(
        self,
        agent,
        env,
        eval_env,
        bench_params,
        eval_env_params,
        eval_interval: int,
        n_eval_episodes: int,
        best_model_save_path: str,
        result_path: str,
        n_cpus: int = 1,
        env_norm: bool = False,
        **kwargs,
    ):
        super(EvalCallback, self).__init__(**kwargs)
        self.env = env
        self.eval_env = eval_env
        self.eval_interval = eval_interval
        self.n_eval_episodes = n_eval_episodes
        self.best_mean_reward = -np.inf
        self.best_model_save_path = best_model_save_path
        self.result_path = result_path
        self.evaluator = None
        self.bench_params = bench_params
        self.eval_env_params = eval_env_params
        if bench_params["discrete_action"]:
            if isinstance(eval_env.action_choices[0][0], list):
                self.evaluator = MultiParamEval(
                    agent=agent,
                    env=self.env,
                    eval_env=self.eval_env,
                    bench_params=self.bench_params,
                    eval_env_params=self.eval_env_params,
                    n_eval_episodes_per_instance=self.n_eval_episodes,
                    log_path=f"{self.best_model_save_path}",
                    n_cpus=n_cpus,
                    env_norm=env_norm,
                )
            else:
                self.evaluator = SingleParamEval(
                    agent=agent,
                    eval_env=self.eval_env,
                    bench_params=self.bench_params,
                    eval_env_params=self.eval_env_params,
                    n_eval_episodes_per_instance=self.n_eval_episodes,
                    log_path=f"{self.best_model_save_path}",
                    n_cpus=n_cpus,
                )
        else:
            if len(bench_params["action_bounds"]) == 1:
                ## single param continuous action
                self.evaluator = SingleParamEval(
                    agent=agent,
                    eval_env=self.eval_env,
                    bench_params=self.bench_params,
                    eval_env_params=self.eval_env_params,
                    n_eval_episodes_per_instance=self.n_eval_episodes,
                    log_path=f"{self.best_model_save_path}",
                    n_cpus=n_cpus,
                )
            else:
                ## multi param continuous action
                self.evaluator = MultiParamEval(
                    agent=agent,
                    env=env,
                    eval_env=self.eval_env,
                    bench_params=self.bench_params,
                    eval_env_params=self.eval_env_params,
                    n_eval_episodes_per_instance=self.n_eval_episodes,
                    log_path=f"{self.best_model_save_path}",
                    n_cpus=n_cpus,
                    env_norm=env_norm,
                )

    def _on_step(self) -> bool:
        """
        This method will be called in the model's `learn` method.
        We evaluate the agent every `eval_interval` steps.
        """
        if (self.n_calls + 1) % self.eval_interval == 0:
            self.evaluator.eval(self.n_calls + 1)
        return True


def test_agent(
    bench_params, eval_env_params, out_dir: str, topk: int = 5, n_cpus: int = 1
):
    eval_data = np.load(os.path.join(out_dir, "evaluations.npz"), allow_pickle=True)
    # mean runtime of learnt policies
    eval_runtime_means = [ls[0] for ls in eval_data["eval_runtime_means"]]
    # best policy
    eval_policies = np.array(
        eval_data["eval_policies"]
    )  # shape (total_steps//eval_interval, instance_idx, policy)
    eval_runtime_means = np.array([ls[0] for ls in eval_data["eval_runtime_means"]])
    top_k_min_indices = np.argsort(eval_runtime_means)[:topk]
    runtime_means = []
    runtime_stds = []
    policies = []
    steps = []
    eval_runtimes = []
    for step in top_k_min_indices:
        policy = eval_policies[step][0]
        if len(policy.shape) > 1:
            runtimes = Parallel(n_jobs=n_cpus)(
                delayed(ollga_multi_param)(bench_params, eval_env_params, policy, i)
                for i in tqdm(
                    range(1000),
                    desc=f"[Test Stage]: Progress Policy @ {step}-th",
                    disable=False,
                )
            )
        else:
            runtimes = Parallel(n_jobs=n_cpus)(
                delayed(ollga_single_param)(bench_params, eval_env_params, policy, i)
                for i in tqdm(
                    range(1000),
                    desc=f"[Test Stage]: Progress Policy @ {step}-th",
                    disable=False,
                )
            )
        eval_runtimes.append(runtimes)
        runtime_means.append(np.mean(runtimes))
        runtime_stds.append(np.std(runtimes))
        policies.append(policy)
        steps.append(step)
    ## print the best runtime
    best_idx = np.argmin(runtime_means)
    # print(f"Best runtime: {runtime_means[best_idx]} +/- {runtime_stds[best_idx]}")
    ## save runtimes
    np.savez(
        os.path.join(out_dir, "evaluations_last.npz"),
        eval_runtime_means=runtime_means,
        eval_runtime_stds=runtime_stds,
        eval_policies=policies,
        eval_runtimes=eval_runtimes,
    )
    return best_idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir", "-o", type=str, default="outputs", help="output folder"
    )
    parser.add_argument(
        "--setting-file", "-s", type=str, help="yml file with all settings"
    )
    parser.add_argument(
        "--seed", type=int, default=1, help="random seed for reproducibility"
    )
    parser.add_argument(
        "--ent-coef", "-e", type=float, default=0.0, help="entropy coefficient"
    )
    parser.add_argument(
        "--n-cpus",
        "-c",
        type=int,
        default=1,
        help="number of CPUs to use for evaluation",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="batch size for training"
    )
    parser.add_argument(
        "--clip-range", type=float, default=0.2, help="clipping range for PPO"
    )
    parser.add_argument(
        "--gae-lambda", type=float, default=0.95, help="GAE lambda parameter"
    )
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument(
        "--learning-rate", "--lr", type=float, default=0.0003, help="learning rate"
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=10,
        help="number of epochs for policy optimization",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=2048,
        help="number of steps to run for each environment per update",
    )
    parser.add_argument(
        "--env-norm",
        action="store_true",
        help="normalize the rewards during training",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=200000,
        help="total timesteps for training",
    )
    parser.add_argument(
        "--target-kl", type=float, default=None, help="target KL divergence"
    )
    args = parser.parse_args()

    # Get configuration from train_conf_ppo.yml
    config_yml_fn = args.setting_file
    (
        exp_params,
        bench_params,
        agent_params,
        train_env_params,
        eval_env_params,
    ) = read_config(config_yml_fn)

    date_str, time_str = get_time_str()

    # Define defaults to check against
    DEFAULTS = {
        "batch_size": 64,
        "clip_range": 0.2,
        "gae_lambda": 0.95,
        "gamma": 0.99,
        "learning_rate": 0.0003,
        "n_epochs": 10,
        "n_steps": 2048,
        "ent_coef": 0.0,
        "env_norm": False,
        "total_timesteps": exp_params["n_steps"],
        "target_kl": None,
    }

    batch_size = args.batch_size
    clip_range = args.clip_range
    gae_lambda = args.gae_lambda
    gamma = args.gamma
    learning_rate = args.learning_rate
    policy_kwargs = dict(
        squash_output=True if not bench_params["discrete_action"] else False,
        log_std_init=-1 if not bench_params["discrete_action"] else None,
    )
    use_sde = True if not bench_params["discrete_action"] else False
    ent_coef = args.ent_coef
    n_epochs = args.n_epochs
    n_steps = args.n_steps
    n_cpus = args.n_cpus
    env_norm = args.env_norm
    total_timesteps = args.total_timesteps
    target_kl = args.target_kl

    # Check which parameters were overridden and build suffix
    overridden_params = []
    param_mapping = {
        "batch_size": batch_size,
        "clip_range": clip_range,
        "gae_lambda": gae_lambda,
        "gamma": gamma,
        "learning_rate": learning_rate,
        "n_epochs": n_epochs,
        "n_steps": n_steps,
        "ent_coef": ent_coef,
        "env_norm": env_norm,
        "total_timesteps": total_timesteps,
        "target_kl": target_kl,
    }

    for param, current_val in param_mapping.items():
        if current_val != DEFAULTS[param]:
            # Format the parameter name and value for directory naming
            param_short = param.replace("_", "").replace("learning_rate", "lr")
            overridden_params.append(f"{param_short}{current_val}")

    # Create directory suffix from overridden parameters
    override_suffix = "_".join(overridden_params) if overridden_params else "default"

    exp_name = config_yml_fn.split("/")[-1].split(".")[0]
    out_dir = os.path.join(
        args.out_dir, f"{exp_name}/{override_suffix}/{date_str}/{time_str}_{args.seed}"
    )
    if os.path.isdir(out_dir) is False:
        os.makedirs(out_dir, exist_ok=True)
        shutil.copyfile(args.setting_file, os.path.join(out_dir, "config.yml"))
        ## dump the config to the output folder
        with open(os.path.join(out_dir, "train_args.yml"), "w") as f:
            yaml.dump(
                {
                    "batch_size": batch_size,
                    "clip_range": clip_range,
                    "gae_lambda": gae_lambda,
                    "gamma": gamma,
                    "learning_rate": learning_rate,
                    "ent_coef": ent_coef,
                    "n_epochs": n_epochs,
                    "policy_kwargs": policy_kwargs,
                    "use_sde": use_sde,
                    "n_steps": n_steps,
                    "env_norm": env_norm,
                    "total_timesteps": total_timesteps,
                    "target_kl": target_kl,
                },
                f,
            )
    # env = make_env(bench_params, train_env_params)
    # Normalize the rewards
    # env = gym.wrappers.NormalizeReward(env)
    # Write results to a log file
    # env = Monitor(env, os.path.join(out_dir, f"monitor_{args.seed}"))

    # Training environment
    # env = DummyVecEnv([env])
    if env_norm:
        if "rescale_action" in bench_params:
            if "independent_action" in bench_params:
                print("Using independent action rescaling with VecNormalize")
                
                env = DummyVecEnv(
                    [
                        lambda: Monitor(
                            NormalizeActionWrapper(
                                make_env(bench_params, train_env_params),
                                independent=bench_params["independent_action"],
                            ),
                            os.path.join(out_dir, f"monitor_{args.seed}"),
                        )
                    ]
                )
            else:
                print("Using action rescaling with VecNormalize")
                
                env = DummyVecEnv(
                    [
                        lambda: Monitor(
                            NormalizeActionWrapper(
                                make_env(bench_params, train_env_params)
                            ),
                            os.path.join(out_dir, f"monitor_{args.seed}"),
                        )
                    ]
                )
        else:
            env = DummyVecEnv(
                [
                    lambda: Monitor(
                        make_env(bench_params, train_env_params),
                        os.path.join(out_dir, f"monitor_{args.seed}"),
                    )
                ]
            )
        env = VecNormalize(env, norm_reward=True, norm_obs=True, gamma=gamma)
    else:
        env = make_env(bench_params, train_env_params)
        env = Monitor(env, os.path.join(out_dir, f"monitor_{args.seed}"))
        if "rescale_action" in bench_params:
            if "independent_action" in bench_params:
                print("Using independent action normalization without VecNormalize")
                env = NormalizeActionWrapper(
                    env, independent=bench_params["independent_action"]
                )
            else:
                env = NormalizeActionWrapper(env)

    # Create the evaluation environment
    eval_env = make_env(bench_params, eval_env_params)

    # PPO agent
    if agent_params["name"] == "ppo":
        model = PPO(
            "MlpPolicy",
            env,
            verbose=0,
            batch_size=batch_size,
            clip_range=clip_range,
            gae_lambda=gae_lambda,
            gamma=gamma,
            learning_rate=learning_rate,
            ent_coef=ent_coef,
            n_epochs=n_epochs,
            policy_kwargs=policy_kwargs,
            use_sde=use_sde,
            n_steps=n_steps,
            target_kl=target_kl,
        )
        # Use the custom callback to evaluate agent's performance after a certain number of steps
        eval_callback = EvalCallback(
            agent=model,
            env=env,
            eval_env=eval_env,
            bench_params=bench_params,
            eval_env_params=eval_env_params,
            eval_interval=exp_params["eval_interval"],
            n_eval_episodes=exp_params["eval_n_episodes"],
            best_model_save_path=out_dir,
            result_path=os.path.join(out_dir, "eval_infos.gzip"),
            n_cpus=n_cpus,
            env_norm=env_norm,
        )

        callback_list = [
            eval_callback,
        ]

        # Train the agent and pass custom callback
        print(f"Training PPO agent for {total_timesteps} timesteps...")
        model.learn(total_timesteps=total_timesteps, callback=callback_list)
        # Save the model
        model.save(os.path.join(out_dir, "ppo_final"))
    else:
        print(f"Sorry, agent {agent_params['name']} is not yet supported")

if __name__ == "__main__":
    main()
