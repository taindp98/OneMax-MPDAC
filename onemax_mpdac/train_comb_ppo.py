import argparse
import os
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from onemax_mpdac.utils import make_env, read_config, get_time_str
import shutil
import numpy as np
import yaml
from joblib import Parallel, delayed
from tqdm import tqdm
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from onemax_mpdac.eval import OneMaxCombEval, ollga_multi_param, ollga_single_param

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
        out_dir: str = "outputs",
        **kwargs,
    ):
        super(EvalCallback, self).__init__(**kwargs)
        self.agent = agent
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
        self.evaluator = OneMaxCombEval(
            self,
            obs_space=self.eval_env.observation_space.shape[0],
            n_eval_episodes_per_instance=n_eval_episodes,
            log_path=os.path.join(best_model_save_path, "eval"),
            n_cpus=n_cpus,
        )
        print(
            "Action space dimension:", agent.policy.action_space.n
        )  # Number of actions

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
    eval_data = np.load(
        os.path.join(out_dir, "eval", "evaluations.npz"), allow_pickle=True
    )
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
        os.path.join(out_dir, "eval", "evaluations_last.npz"),
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

    # if exp_params["n_cores"] > 1:
    #     print("WARNING: n_cores>1 is not yet supported")

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

    ## Normalize the environment (both observations and rewards)
    env = DummyVecEnv(
        [
            lambda: Monitor(
                make_env(bench_params, train_env_params),
                os.path.join(out_dir, f"monitor_{args.seed}"),
            )
        ]
    )
    env = VecNormalize(env, norm_reward=True, norm_obs=True, gamma=gamma)

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
