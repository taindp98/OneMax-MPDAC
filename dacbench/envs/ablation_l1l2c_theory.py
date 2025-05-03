import gymnasium as gym
import numpy as np

from dacbench.envs.theory import OLLGATheoryEnv


class OLLGAFactL1L2CTheoryEnv(OLLGATheoryEnv):
    """
    Environment for (1+(lbd,lbd))-GA with population size.

    Current assumption: we only consider (1+(lbd,lbd))-GAS, so there's only one parameter to tune (lbd)
    """

    def __init__(self, config, test_env=False) -> None:
        """
        Initialize OLLGATheoryEnv.

        Parameters
        ----------
        config : objdict
            Environment configuration
        test_env : bool
            whether to use test mode

        """
        super(OLLGAFactL1L2CTheoryEnv, self).__init__(config)

    def step(self, actions, **kwargs):
        """
        Execute environment step.

        Parameters
        ----------
        action : Box
            action to execute

        Returns
        -------
            state, reward, terminated, truncated, info
            np.array, float, bool, bool, dict

        """
        truncated = super(OLLGATheoryEnv, self).step_()

        fitness_before_update = self.x.fitness

        # get lbd

        # if lbd is out of range
        stop = False

        if stop is False:
            # flip r bits
            mutation_size, crossover_size, beta = actions
            mutation_rate = np.float64(mutation_size / self.n)
            crossover_rate = np.float64(beta / crossover_size)
            ## clip rate to [0,1]
            mutation_rate = np.clip(mutation_rate, 0, 1)
            crossover_rate = np.clip(crossover_rate, 0, 1)
            xprime, f_xprime, ne1 = self.x.mutate(
                p=mutation_rate,
                n_childs=mutation_size,
                rng=self.np_random,
            )
            y, f_y, ne2 = self.x.crossover(
                xprime=xprime,
                p=crossover_rate,
                n_childs=crossover_size,
                rng=self.np_random,
            )
            n_evals = ne1 + ne2
            # update x
            if self.x.fitness <= y.fitness:
                self.x = y

            # update total number of evaluations
            self.total_evals += n_evals

            # check stopping criteria
            terminated = (self.total_evals >= self.max_evals) or (self.x.is_optimal())

            # calculate reward
            if self.reward_choice == "imp_div_evals":
                reward = (self.x.fitness - fitness_before_update - 0.5) / n_evals
            elif self.reward_choice == "imp_minus_evals":
                reward = self.x.fitness - fitness_before_update - n_evals
            elif self.reward_choice == "minus_evals":
                reward = -n_evals
            elif self.reward_choice == "minus_evals_normalised":
                reward = -n_evals / self.max_evals
            elif self.reward_choice == "imp_minus_evals_normalised":
                reward = (
                    self.x.fitness - fitness_before_update - n_evals
                ) / self.max_evals
            elif self.reward_choice == "imp":
                reward = self.x.fitness - fitness_before_update - 0.5
            elif self.reward_choice == "imp_minus_evals_scaling":
                reward = (self.x.fitness - fitness_before_update - n_evals) / self.n
            elif self.reward_choice == "imp_minus_evals_shifting":
                # reward = (self.x.fitness - fitness_before_update - n_evals) - kwargs["shift"]
                reward = self.x.fitness - fitness_before_update - n_evals
                reward -= 3
            elif self.reward_choice == "imp_minus_evals_scaling_shifting":
                reward = (
                    (self.x.fitness - fitness_before_update - n_evals) / self.n
                ) - kwargs["shift"]
            self.log_reward.append(reward)

        # update histories
        self.history_fx.append(self.x.fitness)
        self.history_lbd.append(mutation_size)

        # update logs
        self.log_r.append(mutation_size)
        self.log_fx.append(self.x.fitness)
        self.log_reward.append(reward)

        returned_info = {"msg": "", "values": {}}
        if terminated or truncated:
            if hasattr(self, "env_type"):
                msg = "Env " + self.env_type + ". "
            else:
                msg = ""
            msg += (
                "Episode done: n=%d; obj=%d; init_obj=%d; evals=%d; max_evals=%d; steps=%d; r_min=%.1f; r_max=%.1f; r_mean=%.1f; R=%.4f"
                % (
                    self.n,
                    self.x.fitness,
                    self.init_obj,
                    self.total_evals,
                    self.max_evals,
                    self.c_step,
                    min(self.log_r),
                    max(self.log_r),
                    sum(self.log_r) / len(self.log_r),
                    sum(self.log_reward),
                )
            )
            # self.logger.info(msg)
            returned_info["msg"] = msg
            returned_info["values"] = {
                "n": int(self.n),
                "obj": int(self.x.fitness),
                "init_obj": int(self.init_obj),
                "evals": int(self.total_evals),
                "max_evals": int(self.max_evals),
                "steps": int(self.c_step),
                "r_min": float(min(self.log_r)),
                "r_max": float(max(self.log_r)),
                "r_mean": float(sum(self.log_r) / len(self.log_r)),
                "R": float(sum(self.log_reward)),
                "log_r": [int(x) for x in self.log_r],
                "log_fx": [int(x) for x in self.log_fx],
                "log_reward": [float(x) for x in self.log_reward],
            }

        return self.get_state(), reward, truncated, terminated, returned_info


class OLLGATheoryL1L2CEnvDiscrete(OLLGATheoryEnv):
    """OLLGA environment where the choices of lambda is discretised."""

    def __init__(self, config, test_env=False):
        """Init env."""
        super(OLLGATheoryL1L2CEnvDiscrete, self).__init__(config, test_env)
        assert (
            "action_choices" in config
        ), "Error: action_choices must be specified in benchmark's config"
        assert isinstance(
            self.action_space, gym.spaces.Discrete
        ), "Error: action space must be discrete"
        assert self.action_space.n == len(config["action_choices"][0]), (
            "Error: action space's size (%d) must be equal to the len(action_choices) (%d)"
            % (self.action_space.n, len(config["action_choices"][0]))
        )
        self.discrete_action = True
        self.action_choices = config["action_choices"]

    def step(self, action, **kwargs):
        """Take step."""
        if isinstance(action, np.ndarray) or isinstance(action, list):
            assert len(action) == 1
            action = action[0]
        action_value = self.action_choices[self.inst_id][action]
        return super(OLLGATheoryL1L2CEnvDiscrete, self).step(action_value, **kwargs)


class OLLGAFactL1L2CTheoryEnvDiscrete(OLLGAFactL1L2CTheoryEnv):
    """OLLGA environment where the choices of lambda is discretised."""

    def __init__(self, config, test_env=False):
        """Init env."""
        super(OLLGAFactL1L2CTheoryEnvDiscrete, self).__init__(config, test_env)
        assert (
            "action_choices" in config
        ), "Error: action_choices must be specified in benchmark's config"
        assert isinstance(
            self.action_space, gym.spaces.Discrete
        ), "Error: action space must be discrete"
        assert self.action_space.n == len(config["action_choices"][0][0]), (
            "Error: action space's size (%d) must be equal to the len(action_choices) (%d)"
            % (self.action_space.n, len(config["action_choices"][0][0]))
        )
        self.discrete_action = True
        self.action_choices = config["action_choices"]

    def step(self, actions, **kwargs):
        """Take step."""
        lbd1_idx, lbd2_idx, crossover_idx = actions
        action_value = [
            self.action_choices[self.inst_id][0][lbd1_idx],
            self.action_choices[self.inst_id][1][lbd2_idx],
            self.action_choices[self.inst_id][2][crossover_idx],
        ]
        return super(OLLGAFactL1L2CTheoryEnvDiscrete, self).step(action_value, **kwargs)
