import os
from typing import Optional
import numpy as np
from joblib import Parallel, delayed
from onemax_mpdac.utils import make_env
from tqdm import tqdm


def ollga_single_param(
    bench_params,
    eval_env_params,
    policy,
    seed: int,
):
    """
    Single parameter control of (1+LL)-GA
    Args:
        bench_params: configurations of benchmark
        eval_env_params: configurations of algorithm environment
        policy: list of chosen actions (lambda only) for all states
        seed: random seed
    """
    bench_params["seed"] = seed
    env = make_env(bench_params, eval_env_params)
    instance_set = env.instance_set
    inst = instance_set[0]
    n = inst["size"]
    f_x = env.x.fitness
    total_evals = 1
    cutoff = 0.8 * n * n
    while True:
        mutation_size = crossover_size = policy[f_x]
        mutation_rate = mutation_size / n
        crossover_rate = 1 / crossover_size
        ## mutation phase
        xprime, f_xprime, ne1 = env.x.mutate(
            p=mutation_rate,
            n_childs=mutation_size,
            rng=env.np_random,
        )
        ## crossover phase
        y, f_y, ne2 = env.x.crossover(
            xprime=xprime,
            p=crossover_rate,
            n_childs=crossover_size,
            rng=env.np_random,
        )
        ## selection phase
        if f_x <= f_y:
            f_x = f_y
            env.x = y
        total_evals = total_evals + ne1 + ne2
        if total_evals >= cutoff or env.x.is_optimal():
            break
    return total_evals


def ollga_multi_param(
    bench_params,
    eval_env_params,
    policy,
    seed: int,
):
    """
    Multiple parameters control of (1+LL)-GA
    Args:
        bench_params: configurations of benchmark
        eval_env_params: configurations of algorithm environment
        policy: list of chosen actions (lambda_m, alpha, lambda_c, beta) for all states
        seed: random seed
    """
    bench_params["seed"] = seed
    env = make_env(bench_params, eval_env_params)
    instance_set = env.instance_set
    inst = instance_set[0]
    n = inst["size"]
    f_x = env.x.fitness
    total_evals = 1
    cutoff = 0.8 * n * n
    while True:
        mutation_size, alpha, crossover_size, gamma = policy[f_x]
        ## mutation phase
        mutation_rate = alpha * mutation_size / n
        crossover_rate = gamma / crossover_size
        ## clip crossover_rate to [0, 1]
        mutation_rate = np.clip(mutation_rate, 0, 1)
        crossover_rate = np.clip(crossover_rate, 0, 1)
        xprime, f_xprime, ne1 = env.x.mutate(
            p=mutation_rate,
            n_childs=mutation_size,
            rng=env.np_random,
        )
        ## crossover phase
        y, f_y, ne2 = env.x.crossover(
            xprime=xprime,
            p=crossover_rate,
            n_childs=crossover_size,
            rng=env.np_random,
        )
        ## selection phase
        if f_x <= f_y:
            f_x = f_y
            env.x = y
        total_evals = total_evals + ne1 + ne2
        if total_evals >= cutoff or env.x.is_optimal():
            break
    return total_evals


class OneMaxFactEval:
    """
    Evaluation module for factored action space representation
    """

    def __init__(
        self,
        agent,
        obs_space,
        n_eval_episodes_per_instance: int = 100,
        log_path: Optional[str] = None,
        save_agent_at_every_eval: bool = False,
        verbose: int = 1,
        name: str = "",
        n_cpus: int = 1,
    ):
        self.eval_env = agent.eval_env
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

        self.obs_space = obs_space
        self.n_eval_episodes_per_instance = n_eval_episodes_per_instance
        self.save_agent_at_every_eval = save_agent_at_every_eval

        # we will calculate optimal policy and its runtime for each instance
        self.instance_set = agent.eval_env.instance_set

        # list of inst_id (keys of self.instance_set)
        self.inst_ids = agent.eval_env.instance_id_list

        # best/last mean_runtime of each instance
        self.best_mean_runtime = [np.inf] * len(self.inst_ids)
        self.last_mean_runtime = [np.inf] * len(self.inst_ids)

        # element i^th: optimal policy for instance self.inst_ids[i]
        self.optimal_policies = []
        self.optimal_runtime_means = []
        self.optimal_runtime_stds = []

        # evaluation timesteps
        self.eval_timesteps = []

        self.eval_policies = []
        self.eval_policies_unclipped = []
        self.eval_runtime_means = []
        self.eval_runtime_stds = []
        self.eval_n_decisions = []

        if hasattr(agent.eval_env, "action_choices"):
            self.action_choices = agent.eval_env.action_choices
            self.discrete_portfolio = True
        else:
            self.discrete_portfolio = False

        self.bench_params = agent.bench_params
        self.eval_env_params = agent.eval_env_params
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
                    for k in sorted(
                        agent.eval_env.action_choices[inst_id][0], reverse=True
                    )
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
            print(f"Theory ERT: {runtime_mean} +/- {runtime_std}")
            self.optimal_runtime_means.append(runtime_mean)
            self.optimal_runtime_stds.append(runtime_std)

            if self.verbose >= 1:
                print(
                    f"\t[env: {self.name}] instance: {inst_id}. Runtime: {runtime_mean} +/- {runtime_std}"
                )
            if self.verbose >= 2:
                print("\t" + " ".join([str(v) for v in policy]))

    def eval(self, n_steps) -> bool:
        if self.verbose >= 1:
            print(f"steps: {n_steps}")

        self.eval_timesteps.append(n_steps)

        policies = []
        policies_unclipped = []
        runtime_means = []
        runtime_stds = []

        for inst_id in self.inst_ids:
            inst = self.instance_set[inst_id]
            n = inst["size"]

            policy_unclipped = self.agent.get_actions_for_all_states(n)
            if self.discrete_portfolio:
                policy = []
                for fitness, sel in enumerate(policy_unclipped):
                    """
                    bench_params["name"] distinguishes the set of controlled parameters
                        OLLGAFactTheory: controls 4 parameters
                        OLLGAFactL1L2Theory: controls lambda_m and lambda_c
                        OLLGAFactL1L2MTheory: controls lambda_m, alpha, and lambda_c
                        OLLGAFactL1L2CTheory: controls lambda_m, lambda_c, and beta
                        OLLGAFactL1Theory: controls lambda_m alone
                        OLLGAFactL1MTheory: controls lambda_m and alpha
                        OLLGAFactL1CTheory: controls lambda_m and beta
                        OLLGAFactL1MCTheory: controls lambda_m, alpha and beta
                    """
                    if self.bench_params["name"] == "OLLGAFactTheory":
                        lbd1_idx, mr_idx, lbd2_idx, cr_idx = sel
                        lambda1 = self.action_choices[inst_id][0][lbd1_idx]
                        mutation_rate = self.action_choices[inst_id][1][mr_idx]
                        lambda2 = self.action_choices[inst_id][2][lbd2_idx]
                        crossover_rate = self.action_choices[inst_id][3][cr_idx]
                    elif self.bench_params["name"] == "OLLGAFactL1L2Theory":
                        lbd1_idx, lbd2_idx = sel
                        lambda1 = self.action_choices[inst_id][0][lbd1_idx]
                        mutation_rate = 1
                        lambda2 = self.action_choices[inst_id][1][lbd2_idx]
                        crossover_rate = 1
                    elif self.bench_params["name"] == "OLLGAFactL1L2MTheory":
                        lbd1_idx, mutation_idx, lbd2_idx = sel
                        lambda1 = self.action_choices[inst_id][0][lbd1_idx]
                        mutation_rate = self.action_choices[inst_id][1][mutation_idx]
                        lambda2 = self.action_choices[inst_id][2][lbd2_idx]
                        crossover_rate = 1
                    elif self.bench_params["name"] == "OLLGAFactL1L2CTheory":
                        lbd1_idx, lbd2_idx, cr_idx = sel
                        lambda1 = self.action_choices[inst_id][0][lbd1_idx]
                        mutation_rate = 1
                        lambda2 = self.action_choices[inst_id][1][lbd2_idx]
                        crossover_rate = self.action_choices[inst_id][2][cr_idx]
                    elif self.bench_params["name"] == "OLLGAFactL1Theory":
                        lambda1 = self.action_choices[inst_id][0][sel[0]]
                        mutation_rate = 1
                        lambda2 = lambda1
                        crossover_rate = 1
                    elif self.bench_params["name"] == "OLLGAFactL1MTheory":
                        lbd1_idx, mutation_idx = sel
                        lambda1 = self.action_choices[inst_id][0][lbd1_idx]
                        mutation_rate = self.action_choices[inst_id][1][mutation_idx]
                        lambda2 = lambda1
                        crossover_rate = 1
                    elif self.bench_params["name"] == "OLLGAFactL1CTheory":
                        lbd1_idx, crossover_idx = sel
                        lambda1 = self.action_choices[inst_id][0][lbd1_idx]
                        crossover_rate = self.action_choices[inst_id][1][crossover_idx]
                        lambda2 = lambda1
                        mutation_rate = 1
                    elif self.bench_params["name"] == "OLLGAFactL1MCTheory":
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

            policies.append(policy)
            policies_unclipped.append(policy_unclipped)

            # calculate runtime of current policy
            if self.obs_space == 2:
                ## Since we know all states, we can parallelize the optimization execution
                self.eval_env.instance_id_list = [inst_id]
                self.eval_env.instance_index = 0
                self.eval_env.instance_set = {inst_id: inst}
                runtimes = Parallel(n_jobs=self.n_cpus)(
                    delayed(ollga_multi_param)(
                        self.bench_params, self.eval_env_params, policy, i
                    )
                    for i in range(self.n_eval_episodes_per_instance)
                )

                runtime_mean = np.mean(runtimes)
                runtime_std = np.std(runtimes)

                self.eval_env.instance_id_list = self.inst_ids
                self.eval_env.instance_set = self.instance_set
            else:
                ## Since we dont know all states, we need to call Q network and predict the action for every state
                episode_rewards = []
                for ep_id in range(self.n_eval_episodes_per_instance):
                    s, _ = self.eval_env.reset()
                    ep_r = 0  # episode's total reward
                    d = False
                    cutoff = 0.8 * n * n
                    while True:
                        actions, _ = self.agent.policy.predict([s])
                        ns, r, tr, d, _ = self.eval_env.step(actions[0])
                        ep_r += r
                        if d or ep_r >= cutoff:
                            break
                    episode_rewards.append(abs(ep_r))
                runtime_mean = np.mean(episode_rewards)
                runtime_std = np.std(episode_rewards)
            runtime_means.append(runtime_mean)
            runtime_stds.append(runtime_std)

            if self.verbose >= 1:
                print(
                    f"\t[env: {self.name}] instance: {inst_id}. Runtime: {runtime_mean} +/- {runtime_std}"
                )
            if self.verbose >= 2:
                print("\t" + " ".join([str(v) for v in policy]))

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
                eval_policies_unclipped=np.array(
                    self.eval_policies_unclipped, dtype=object
                ),
                eval_runtime_means=self.eval_runtime_means,
                eval_runtime_stds=self.eval_runtime_stds,
                instance_set=self.instance_set,
            )
            # save current model
            if self.save_agent_at_every_eval:
                self.agent.save_model(os.path.join(self.log_path, f"model_{n_steps}"))

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
                if self.verbose >= 1:
                    print(
                        f"\t[env: {self.name}] New best mean runtime! ({runtime_means})"
                    )
                if self.log_path:
                    self.agent.save_model(os.path.join(self.log_path, "best_model"))
        return runtime_means, runtime_stds


class OneMaxCombEval:
    """
    Evaluation module for combinatorial action space representation
    """

    def __init__(
        self,
        agent,
        obs_space,
        n_eval_episodes_per_instance: int = 100,
        log_path: Optional[str] = None,
        save_agent_at_every_eval: bool = False,
        verbose: int = 1,
        name: str = "",
        n_cpus: int = 1,
    ):
        self.eval_env = agent.eval_env
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

        self.obs_space = obs_space
        self.n_eval_episodes_per_instance = n_eval_episodes_per_instance
        self.save_agent_at_every_eval = save_agent_at_every_eval

        # we will calculate optimal policy and its runtime for each instance
        self.instance_set = agent.eval_env.instance_set

        # list of inst_id (keys of self.instance_set)
        self.inst_ids = agent.eval_env.instance_id_list

        # best/last mean_runtime of each instance
        self.best_mean_runtime = [np.inf] * len(self.inst_ids)
        self.last_mean_runtime = [np.inf] * len(self.inst_ids)

        # element i^th: optimal policy for instance self.inst_ids[i]
        self.optimal_policies = []
        self.optimal_runtime_means = []
        self.optimal_runtime_stds = []

        # evaluation timesteps
        self.eval_timesteps = []
        self.eval_policies = []
        self.eval_policies_unclipped = []
        self.eval_runtime_means = []
        self.eval_runtime_stds = []
        self.eval_n_decisions = []

        if hasattr(agent.eval_env, "action_choices"):
            self.action_choices = agent.eval_env.action_choices
            self.discrete_portfolio = True
        else:
            self.discrete_portfolio = False

        self.bench_params = agent.bench_params
        self.eval_env_params = agent.eval_env_params
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
                action_choices = np.array(agent.eval_env.action_choices)
                lbd1_portfolio = action_choices[:, 0]
                ## get unique values
                lbd1_portfolio = np.unique(lbd1_portfolio).astype(int)
                portfolio = [k for k in sorted(lbd1_portfolio, reverse=True) if k < n]
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
            print(f"Theory ERT: {runtime_mean} +/- {runtime_std}")
            self.optimal_runtime_means.append(runtime_mean)
            self.optimal_runtime_stds.append(runtime_std)

            if self.verbose >= 1:
                print(
                    f"\t[env: {self.name}] instance: {inst_id}. Runtime: {runtime_mean} +/- {runtime_std}"
                )
            if self.verbose >= 2:
                print("\t" + " ".join([str(v) for v in policy]))

    def eval(self, n_steps) -> bool:
        if self.verbose >= 1:
            print(f"steps: {n_steps}")

        self.eval_timesteps.append(n_steps)

        policies = []
        policies_unclipped = []
        runtime_means = []
        runtime_stds = []

        for inst_id in self.inst_ids:
            inst = self.instance_set[inst_id]
            n = inst["size"]

            policy_unclipped = self.agent.get_actions_for_all_states(n)
            if self.discrete_portfolio:
                policy = []
                for fitness, sel in enumerate(policy_unclipped):
                    lambda1, mutation_rate, lambda2, crossover_rate = (
                        self.action_choices[sel]
                    )
                    policy.append(
                        [
                            np.int64(lambda1),
                            np.float64(mutation_rate),
                            np.int64(lambda2),
                            np.float64(crossover_rate),
                        ]
                    )

            policies.append(policy)
            policies_unclipped.append(policy_unclipped)

            # calculate runtime of current policy
            if self.obs_space == 2:
                ## Since we know all states, we can parallelize the optimization execution
                self.eval_env.instance_id_list = [inst_id]
                self.eval_env.instance_index = 0
                self.eval_env.instance_set = {inst_id: inst}
                runtimes = Parallel(n_jobs=self.n_cpus)(
                    delayed(ollga_multi_param)(
                        self.bench_params, self.eval_env_params, policy, i
                    )
                    for i in range(self.n_eval_episodes_per_instance)
                )

                runtime_mean = np.mean(runtimes)
                runtime_std = np.std(runtimes)

                self.eval_env.instance_id_list = self.inst_ids
                self.eval_env.instance_set = self.instance_set
            else:
                ## Since we dont know all states, we need to call Q network and predict the action for every state
                raise NotImplementedError

            runtime_means.append(runtime_mean)
            runtime_stds.append(runtime_std)

            if self.verbose >= 1:
                print(
                    f"\t[env: {self.name}] instance: {inst_id}. Runtime: {runtime_mean} +/- {runtime_std}"
                )
            if self.verbose >= 2:
                print("\t" + " ".join([str(v) for v in policy]))

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
                eval_policies_unclipped=np.array(
                    self.eval_policies_unclipped, dtype=object
                ),
                eval_runtime_means=self.eval_runtime_means,
                eval_runtime_stds=self.eval_runtime_stds,
                instance_set=self.instance_set,
            )
            # save current model
            if self.save_agent_at_every_eval:
                self.agent.save_model(os.path.join(self.log_path, f"model_{n_steps}"))

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
                if self.verbose >= 1:
                    print(
                        f"\t[env: {self.name}] New best mean runtime! ({runtime_means})"
                    )
                if self.log_path:
                    self.agent.save_model(os.path.join(self.log_path, "best_model"))
        return runtime_means, runtime_stds
