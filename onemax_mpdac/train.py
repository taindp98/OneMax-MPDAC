import argparse
import os
import gymnasium as gym
import time
from models.factored_ddqn import FactoredDDQN
from models.combinatorial_ddqn import CombinatorialDDQN

from onemax_mpdac.utils import make_env, read_config, seed_everything
import warnings
from torch.nn import functional as F
import torch

# Ignore all warnings
warnings.filterwarnings("ignore")


def main():
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir", "-o", type=str, default="outputs", help="output folder"
    )
    parser.add_argument(
        "--config-file", "-c", type=str, help="yml file with all configs"
    )
    parser.add_argument(
        "--seed", "-s", type=int, help="seed for reproducibility", default=123
    )
    parser.add_argument(
        "--n-cpus", "-n", type=int, help="number of used CPUs", default=1
    )
    parser.add_argument(
        "--gamma", "-g", type=float, help="discount factor", default=0.99
    )

    args = parser.parse_args()

    config_yml_fn = args.config_file
    (
        exp_params,
        bench_params,
        agent_params,
        train_env_params,
        eval_env_params,
    ) = read_config(config_yml_fn)

    # create output folder
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    # create train_env and eval_env
    train_env = make_env(bench_params, train_env_params)
    eval_env = make_env(bench_params, eval_env_params)
    if isinstance(train_env.action_space, gym.spaces.Discrete):
        action_dim = train_env.action_space.n
    else:
        action_dim = train_env.action_space.shape[0]
    state_dim = len(train_env.reset())
    # get loss function
    assert agent_params["loss_function"] in ["mse_loss", "smooth_l1_loss"]
    seed_everything(args.seed)
    if agent_params["name"] == "factored_ddqn":
        agent_class = FactoredDDQN
    elif agent_params["name"] == "combinatorial_ddqn":
        agent_class = CombinatorialDDQN
    else:
        raise ValueError(f"Sorry, agent {agent_params['name']} is not yet supported")

    agent_params["gamma"] = args.gamma
    agent = agent_class(
        state_dim=state_dim,
        action_dim=action_dim,
        env=train_env,
        eval_env=eval_env,
        out_dir=out_dir,
        gamma=agent_params["gamma"],
        lr=agent_params["lr"],
        loss_function=getattr(F, agent_params["loss_function"]),
        seed=args.seed,
        config=config_yml_fn,
        n_cpus=args.n_cpus,
        bench_params=bench_params,
        eval_env_params=eval_env_params,
        net_arch=agent_params["net_arch"],
    )
    agent.train(
        episodes=exp_params["n_episodes"],
        max_env_time_steps=int(1e9),
        epsilon=agent_params["epsilon"],
        eval_every_n_steps=exp_params["eval_interval"],
        save_agent_at_every_eval=exp_params["save_agent_at_every_eval"],
        n_eval_episodes_per_instance=exp_params["n_eval_episodes_per_instance"],
        max_train_time_steps=exp_params["n_steps"],
        begin_learning_after=agent_params["begin_learning_after"],
        batch_size=agent_params["batch_size"],
        log_level=exp_params["log_level"],
    )


if __name__ == "__main__":
    # Set deterministic algorithms
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_default_dtype(torch.float32)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    main()
