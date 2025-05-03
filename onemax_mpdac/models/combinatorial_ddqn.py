import pickle
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple
import time
import sys
import os
import json
from tqdm import tqdm
import shutil
from onemax_mpdac.utils import plot_policies
from onemax_mpdac.loggers import Logger
from onemax_mpdac.utils import load_config, object_to_dict
from joblib import Parallel, delayed
from onemax_mpdac.eval import OneMaxCombEval, ollga_mp_single_run
from onemax_mpdac.utils import get_time_str


def soft_update(target, source, tau):
    """
    Simple Helper for updating target-network parameters
    :param target: target network
    :param source: policy network
    :param tau: weight to regulate how strongly to update (1 -> copy over weights)
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


class Q(nn.Module):
    """
    Simple fully connected Q function.
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        non_linearity=F.relu,
        net_arch: list = [50, 50],
        use_dueling: bool = False,
    ):

        super(Q, self).__init__()
        self.fc1 = nn.Linear(state_dim, net_arch[0])
        self.fc2 = nn.Linear(net_arch[0], net_arch[1])
        self.use_dueling = use_dueling
        if self.use_dueling:
            # Separate value and advantage streams for dueling architecture
            self.value_head = nn.Linear(net_arch[1], 1)
            self.advantage_head = nn.Linear(net_arch[1], action_dim)
        else:
            # Single output layer for standard Q-network
            self.fc3 = nn.Linear(net_arch[1], action_dim)
        self._non_linearity = non_linearity

    def forward(self, x):
        x = self._non_linearity(self.fc1(x))
        x = self._non_linearity(self.fc2(x))
        if self.use_dueling:
            # Compute value and advantage streams
            value = self.value_head(x)
            advantage = self.advantage_head(x)
            # Combine value and advantage streams to compute Q-values
            q_val = value + advantage - advantage.mean(dim=1, keepdim=True)
        else:
            # Standard Q-network forward pass
            q_val = self.fc3(x)
        return q_val


class ReplayBuffer:
    """
    Simple Replay Buffer. Used for standard DQN learning.
    """

    def __init__(self, max_size, rng=np.random.default_rng()):
        self._data = namedtuple(
            "ReplayBuffer",
            ["states", "actions", "next_states", "rewards", "terminal_flags"],
        )
        self._data = self._data(
            states=[], actions=[], next_states=[], rewards=[], terminal_flags=[]
        )
        self._size = 0
        self._max_size = max_size
        self.rng = rng
        # self.reward_scaler = RewardScaler()

    def add_transition(self, state, action, next_state, reward, done):
        self._data.states.append(state)
        self._data.actions.append(action)
        self._data.next_states.append(next_state)

        self._data.rewards.append(reward)
        self._data.terminal_flags.append(done)
        self._size += 1
        if self._size > self._max_size:
            self._data.states.pop(0)
            self._data.actions.pop(0)
            self._data.next_states.pop(0)
            self._data.rewards.pop(0)
            self._data.terminal_flags.pop(0)

    def random_next_batch(self, batch_size):
        batch_indices = self.rng.choice(len(self._data.states), batch_size)
        batch_states = np.array([self._data.states[i] for i in batch_indices])
        batch_actions = np.array([self._data.actions[i] for i in batch_indices])
        batch_next_states = np.array([self._data.next_states[i] for i in batch_indices])
        batch_rewards = np.array([self._data.rewards[i] for i in batch_indices])

        batch_terminal_flags = np.array(
            [self._data.terminal_flags[i] for i in batch_indices]
        )
        return (
            batch_states,
            batch_actions,
            batch_next_states,
            batch_rewards,
            batch_terminal_flags,
        )

    def save(self, path):
        with open(os.path.join(path, "rpb.pkl"), "wb") as fh:
            pickle.dump(list(self._data), fh)

    def load(self, path):
        with open(os.path.join(path, "rpb.pkl"), "rb") as fh:
            data = pickle.load(fh)
        self._data = namedtuple(
            "ReplayBuffer",
            ["states", "actions", "next_states", "rewards", "terminal_flags"],
        )
        self._data.states = data[0]
        self._data.actions = data[1]
        self._data.next_states = data[2]
        self._data.rewards = data[3]
        self._data.terminal_flags = data[4]
        self._size = len(data[0])

    def get_reward_stats(self):
        return self._data.rewards


class CombinatorialDDQN:
    """
    Combinatorial Deep Q-Network Agent
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        env: gym.Env,
        eval_env: gym.Env = None,
        extra_eval_env: gym.Env = None,
        out_dir: str = "outputs/checkpoints",
        gamma: float = 0.99,
        loss_function=F.mse_loss,
        lr: float = 1e-3,
        use_double_dqn=True,
        seed: int = 42,
        config: str = None,
        n_cpus=-1,
        bench_params: dict = {},
        eval_env_params: dict = {},
        net_arch: list = [50, 50],
    ):
        """
        Initialize the DQN Agent
        :param state_dim: dimensionality of the input states
        :param action_dim: dimensionality of the output actions
        :param gamma: discount factor
        :param env: environment to train on
        :param eval_env: environment to evaluate on
        :param extra_eval_env: another environment to evaluate on (optional)
        :param vision: boolean flag to indicate if the input state is an image or not
        """
        self.state_dim = state_dim
        self.device = torch.device("cpu")
        self.action_choices = np.array(env.action_choices)
        ## compute the total number of possible actions by combining all possible actions
        self._q = Q(
            state_dim=state_dim,
            action_dim=len(self.action_choices),
            net_arch=net_arch,
        )
        self._q_target = Q(
            state_dim=state_dim,
            action_dim=len(self.action_choices),
            net_arch=net_arch,
        )
        self.seed = seed
        self.rng = np.random.default_rng(seed=seed)
        torch.manual_seed(seed)
        self._gamma = gamma
        self._loss_function = loss_function
        self.lr = lr
        self._q_optimizer = optim.Adam(self._q.parameters(), lr=lr)
        self.use_double_dqn = use_double_dqn
        self.n_cpus = n_cpus
        self._replay_buffer = ReplayBuffer(1e6, self.rng)
        self._env = env
        self.eval_env = eval_env
        self._extra_eval_env = extra_eval_env
        self.bench_params = bench_params
        self.eval_env_params = eval_env_params
        date_str, time_str = get_time_str()
        cfg_name = os.path.basename(config).split(".")[0]
        self.out_dir = os.path.join(
            f"{out_dir}/{date_str}/{cfg_name}/gamma:{gamma}",
            f"{time_str}_seed_{seed}",
        )
        os.makedirs(self.out_dir, exist_ok=True)
        shutil.copyfile(
            config,
            os.path.join(self.out_dir, os.path.basename(config)),
        )
        self.config = object_to_dict(load_config(config))
        self.config["experiment"]["seed"] = seed
        self.config["agent"]["gamma"] = gamma
        self.logger = Logger(
            config=self.config,
            date_str=date_str,
            exp_name=self.__repr__(),
            time_str=time_str,
            seed=seed,
            project=cfg_name,
        )

        self.runtime_logs = {
            "train": [],
            "eval": [],
            "forward": [],
        }
        self.shift_constant = None

    def tt(self, ndarray):
        """
        Helper Function to cast observation to correct type/device
        """
        tensor = torch.tensor(ndarray, dtype=torch.float32)
        return tensor.to(self.device)

    def save_rpb(self, path):
        self._replay_buffer.save(path)

    def load_rpb(self, path):
        self._replay_buffer.load(path)

    def act(self, x: np.ndarray, epsilon: float = 0.0) -> int:
        """
        Simple helper to get action epsilon-greedy based on observation x
        """
        x = self.tt(x)
        # check shape and unsqueeze if necessary
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        u = torch.argmax(self._q(self.tt(x)), dim=1).cpu().numpy()[0]
        r = self.rng.random()
        if r < epsilon:
            return self.rng.integers(low=0, high=len(self.action_choices))
        return u

    def update_epsilon(
        self,
        begin_learning_after,
        max_train_time_steps,
        epsilon_start,
        epsilon_decay_end_point,
        epsilon_decay_end_value,
        cur_total_steps,
    ) -> float:
        """
        Calculate new epsilon value (linearly decay)
        """
        end_step = (
            begin_learning_after
            + (max_train_time_steps - begin_learning_after) * epsilon_decay_end_point
        )

        if cur_total_steps >= end_step:
            return epsilon_decay_end_value

        return epsilon_start - (epsilon_start - epsilon_decay_end_value) * (
            cur_total_steps - begin_learning_after
        ) / (end_step - begin_learning_after)

    def train(
        self,
        episodes: int,
        max_env_time_steps: int = 1_000_000,
        epsilon: float = 0.2,
        epsilon_decay: bool = False,
        epsilon_decay_end_point: float = 0.5,  # ignored if epsilon_decay = False
        epsilon_decay_end_value: float = 0.05,  # ignored if epsilon_decay = False
        eval_every_n_steps: int = 1000,
        n_eval_episodes_per_instance: int = 20,
        save_agent_at_every_eval: bool = True,
        max_train_time_steps: int = 1_000_000,
        begin_learning_after: int = 10_000,
        batch_size: int = 2_048,
        log_level=1,
    ):
        """
        Training loop
        :param episodes: maximum number of episodes to train for
        :param max_env_time_steps: maximum number of steps in the environment to perform per episode
        :param epsilon: constant epsilon for exploration when selecting actions
        :param eval_every_n_steps: interval of steps after which to evaluate the trained agent
        :param max_train_time_steps: maximum number of steps to train
        :param epsilon_decay: whether epsilon decay is enabled. ONLY WORK IF TRAINING BUDGET IS BASED ON max_train_time_steps.
        :param log_level:
            (1) basic log
            (2) extensive log
        :return:
        """
        start_training_time = time.time()
        train_args = {
            "episodes": episodes,
            "max_env_time_steps": max_env_time_steps,
            "epsilon": epsilon,
            "epsilon_decay": epsilon_decay,
            "epsilon_decay_end_point": epsilon_decay_end_point,
            "epsilon_decay_end_value": epsilon_decay_end_value,
            "eval_every_n_steps": eval_every_n_steps,
            "n_eval_episodes_per_instance": n_eval_episodes_per_instance,
            "save_agent_at_every_eval": save_agent_at_every_eval,
            "max_train_time_steps": max_train_time_steps,
            "begin_learning_after": begin_learning_after,
            "batch_size": batch_size,
            "log_level": log_level,
        }
        self.config["train_args"] = train_args
        with open(os.path.join(self.out_dir, "train_args.json"), "w") as fh:
            json.dump(self.config, fh, indent=4)

        total_steps = 0

        self.evaluator = OneMaxCombEval(
            self,
            obs_space=self.eval_env.observation_space.shape[0],
            n_eval_episodes_per_instance=n_eval_episodes_per_instance,
            log_path=os.path.join(self.out_dir, "eval"),
            n_cpus=self.n_cpus,
        )

        s = self._env.get_state()

        epsilon_start = epsilon
        pbar = tqdm(total=max_train_time_steps, desc="Training Progress")

        total_rewards = []
        for episode in range(episodes):
            ep_losses = []
            reward_sum = 0
            for t in range(max_env_time_steps):
                if epsilon_decay:
                    epsilon = self.update_epsilon(
                        begin_learning_after,
                        max_train_time_steps,
                        epsilon_start,
                        epsilon_decay_end_point,
                        epsilon_decay_end_value,
                        total_steps,
                    )

                a = self.act(s, epsilon if total_steps > begin_learning_after else 1.0)
                ns, r, tr, d, _ = self._env.step(
                    actions=a,
                    shift=self.shift_constant if self.shift_constant else 0,
                )

                total_steps += 1
                pbar.update(1)
                reward_sum += r

                if total_steps == begin_learning_after:
                    rewards = self._replay_buffer.get_reward_stats()
                    med_rw = np.median(rewards)
                    self.shift_constant = -abs(med_rw) / 5
                    print(f"Adaptive Shifting: {self.shift_constant}")

                if total_steps % eval_every_n_steps == 0:
                    eval_step_runtime_start = time.time()
                    self.evaluator.eval(n_steps=total_steps)
                    self.runtime_logs["eval"].append(
                        {
                            "step": total_steps,
                            "episode": episode,
                            "step_runtime": time.time() - eval_step_runtime_start,
                        }
                    )

                # Update replay buffer
                self._replay_buffer.add_transition(s, a, ns, r, d)
                if total_steps > begin_learning_after:
                    data_batch = self._replay_buffer.random_next_batch(batch_size)
                    train_step_runtime_start = time.time()
                    (
                        batch_states,
                        batch_actions,
                        batch_next_states,
                        batch_rewards,
                        batch_terminal_flags,
                    ) = (
                        self.tt(data_batch[0]),
                        self.tt(data_batch[1]),
                        self.tt(data_batch[2]),
                        self.tt(data_batch[3]),
                        self.tt(data_batch[4]),
                    )
                    ## for MP
                    target = (
                        batch_rewards
                        + (1 - batch_terminal_flags)
                        * self._gamma
                        * self._q_target(batch_next_states)[
                            torch.arange(batch_size).long(),
                            torch.argmax(self._q(batch_next_states), dim=1),
                        ]
                    )

                    current_prediction = self._q(batch_states)[
                        torch.arange(batch_size).long(), batch_actions.long()
                    ]
                    loss = self._loss_function(current_prediction, target.detach())

                    ep_losses.append(loss.item())

                    self._q_optimizer.zero_grad()
                    loss.backward()
                    self._q_optimizer.step()

                    soft_update(self._q_target, self._q, 0.01)

                    self.runtime_logs["train"].append(
                        {
                            "step": total_steps,
                            "episode": episode,
                            "step_runtime": time.time() - train_step_runtime_start,
                        }
                    )
                if d:
                    break
                s = ns
                if total_steps >= max_train_time_steps:
                    break
                if total_steps % eval_every_n_steps == 0:
                    if ep_losses:
                        avg_loss = np.mean(ep_losses)
                        self.logger.log_scalar(
                            tag="Loss/step", value=avg_loss, step=total_steps
                        )

            s, _ = self._env.reset()

            total_rewards.append(reward_sum)

            if total_steps >= max_train_time_steps:
                break

            # Log metrics at the end of each episode
            if ep_losses:
                avg_loss = np.mean(ep_losses)
                self.logger.log_scalar(tag="Loss/episode", value=avg_loss, step=episode)
        # dump runtime logs
        end_training_time = time.time()
        self.runtime_logs["total"] = end_training_time - start_training_time
        with open(
            os.path.join(
                self.out_dir,
                "eval",
                f"runtime_logs_cpus:{self.n_cpus}.json",
            ),
            "w",
        ) as fh:
            json.dump(self.runtime_logs, fh)

        self.test(
            ckpt_dir=self.out_dir,
            verbose=False,
            total_steps=total_steps,
            topk=5,
        )
        # Close the logger
        self.logger.close()

    def __repr__(self):
        return "combinatorial_ddqn"

    def save_model(self, model_path):
        torch.save(self._q.state_dict(), os.path.join(model_path + ".pt"))

    def load(self, q_path):
        self._q.load_state_dict(torch.load(q_path))

    def get_actions_for_all_states(self, n):
        start_time = time.time()
        with torch.no_grad():
            all_states = np.array([[n, fx] for fx in range(0, n)])
            acts = self._q(self.tt(all_states)).argmax(dim=1).cpu().numpy().tolist()
        self.runtime_logs["forward"].append(
            {
                "step_runtime": time.time() - start_time,
            }
        )
        return acts

    def test(
        self,
        ckpt_dir: str,
        topk: int = 5,
        n_cpus: int = 8,
    ):
        ## load evaluation data
        evaluations_fpath = os.path.join(ckpt_dir, "eval", "evaluations.npz")
        eval_data = np.load(evaluations_fpath, allow_pickle=True)
        eval_policies = np.array(
            eval_data["eval_policies"]
        )  # shape (total_steps//eval_interval, instance_idx, policy)
        instance_set = eval_data["instance_set"].item()
        i = 0
        inst_id = eval_data["inst_ids"][i]
        instance = instance_set[inst_id]
        n = instance["size"]
        eval_runtime_means = np.array([ls[0] for ls in eval_data["eval_runtime_means"]])
        eval_timesteps = np.array(eval_data["eval_timesteps"])
        top_k_min_indices = np.argsort(eval_runtime_means)[:topk]

        runtime_means = []
        runtime_stds = []
        policies = []
        steps = []
        save_fname = os.path.join(ckpt_dir, "eval", "evaluations_last.npz")
        for step in top_k_min_indices:
            policy = eval_policies[step][0]
            runtimes = Parallel(n_jobs=n_cpus)(
                delayed(ollga_mp_single_run)(
                    self.bench_params, self.eval_env_params, policy, i
                )
                for i in tqdm(
                    range(1000),
                    desc=f"[Test Stage]: Progress Policy @ {step}-th",
                    disable=False,
                )
            )

            runtime_means.append(np.mean(runtimes))
            runtime_stds.append(np.std(runtimes))
            policies.append(policy)
            steps.append(step)

        np.savez(
            file=save_fname,
            n_steps=steps,
            eval_policies=policies,
            eval_runtime_means=runtime_means,
            eval_runtime_stds=runtime_stds,
        )

        plot_policies(results_fpath=os.path.join(ckpt_dir, "eval", "evaluations.npz"))
