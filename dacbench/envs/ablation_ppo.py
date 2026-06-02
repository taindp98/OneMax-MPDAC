import gymnasium as gym
import numpy as np
import random
from gymnasium.utils import seeding
import logging
from dacbench.envs.theory import OneMax, HISTORY_LENGTH
from collections import deque
import uuid


class AbstractEnv(gym.Env):
    """Abstract template for environments."""

    def __init__(self, config):
        """
        Initialize environment.

        Parameters
        ----------
        config : dict
            Environment configuration
            If to seed the action space as well

        """
        super(AbstractEnv, self).__init__()
        self.config = config
        if "instance_update_func" in self.config.keys():
            self.instance_updates = self.config["instance_update_func"]
        else:
            self.instance_updates = "round_robin"
        self.instance_set = config["instance_set"]
        self.instance_id_list = sorted(list(self.instance_set.keys()))
        self.instance_index = 0
        self.inst_id = self.instance_id_list[self.instance_index]
        self.instance = self.instance_set[self.inst_id]

        self.test = False
        if "test_set" in self.config.keys():
            self.test_set = config["test_set"]
            self.test_instance_id_list = sorted(list(self.test_set.keys()))
            self.test_instance_index = 0
            self.test_inst_id = self.test_instance_id_list[self.test_instance_index]
            self.test_instance = self.test_set[self.test_inst_id]

            self.training_set = self.instance_set
            self.training_id_list = self.instance_id_list
            self.training_inst_id = self.inst_id
            self.training_instance = self.instance
        else:
            self.test_set = None

        self.benchmark_info = config["benchmark_info"]
        self.initial_seed = None
        self.np_random = None

        self.n_steps = config["cutoff"]
        self.c_step = 0

        self.reward_range = config["reward_range"]

        if "observation_space" in config.keys():
            self.observation_space = config["observation_space"]
        else:
            if not config["observation_space_class"] == "Dict":
                try:
                    self.observation_space = getattr(
                        gym.spaces, config["observation_space_class"]
                    )(
                        *config["observation_space_args"],
                        dtype=config["observation_space_type"],
                    )
                except KeyError:
                    print(
                        "Either submit a predefined gym.space 'observation_space' or an 'observation_space_class' as well as a list of 'observation_space_args' and the 'observation_space_type' in the configuration."
                    )
                    print("Tuple observation_spaces are currently not supported.")
                    raise KeyError

            else:
                try:
                    self.observation_space = getattr(
                        gym.spaces, config["observation_space_class"]
                    )(*config["observation_space_args"])
                except AssertionError:
                    print(
                        "To use a Dict observation space, the 'observation_space_args' in the configuration should be a list containing a Dict of gym.Spaces"
                    )
                    raise TypeError

        # TODO: use dicts by default for actions and observations
        # The config could change this for RL purposes
        if "config_space" in config.keys():
            actions = config["config_space"].get_hyperparameters()
            action_types = [type(a).__name__ for a in actions]
            # Uniform action space
            if all(t == action_types[0] for t in action_types):
                if "Float" in action_types[0]:
                    low = np.array([a.lower for a in actions])
                    high = np.array([a.upper for a in actions])
                    self.action_space = gym.spaces.Box(low=low, high=high)
                elif "Integer" in action_types[0] or "Categorical" in action_types[0]:
                    if len(action_types) == 1:
                        try:
                            n = actions[0].upper - actions[0].lower
                        except:
                            n = len(actions[0].choices)
                        self.action_space = gym.spaces.Discrete(n**4)
                    else:
                        ns = []
                        for a in actions:
                            try:
                                ns.append(a.upper - a.lower)
                            except:
                                ns.append(len(a.choices))
                        self.action_space = gym.spaces.MultiDiscrete(np.array(ns))
                else:
                    raise ValueError(
                        "Only float, integer and categorical hyperparameters are supported as of now"
                    )
            # Mixed action space
            # TODO: implement this
            else:
                raise ValueError("Mixed type config spaces are currently not supported")
        elif "action_space" in config.keys():
            self.action_space = config["action_space"]
        else:
            try:
                self.action_space = getattr(gym.spaces, config["action_space_class"])(
                    *config["action_space_args"]
                )
            except KeyError:
                print(
                    "Either submit a predefined gym.space 'action_space' or an 'action_space_class' as well as a list of 'action_space_args' in the configuration"
                )
                raise KeyError

            except TypeError:
                print("Tuple and Dict action spaces are currently not supported")
                raise TypeError

        # seeding the environment after initialising action space
        self.seed(config.get("seed", None), config.get("seed_action_space", False))

    def step_(self):
        """
        Pre-step function for step count and cutoff.

        Returns
        -------
        bool
            End of episode

        """
        truncated = False
        self.c_step += 1
        if self.c_step >= self.n_steps:
            truncated = True
        return truncated

    def reset_(self, seed=0, options={}, instance=None, instance_id=None, scheme=None):
        """Pre-reset function for progressing through the instance set.Will either use round robin, random or no progression scheme."""
        if seed is not None:
            self.seed(seed, self.config.get("seed_action_space", False))
        self.c_step = 0
        if scheme is None:
            scheme = self.instance_updates
        self.use_next_instance(instance, instance_id, scheme=scheme)

    def use_next_instance(self, instance=None, instance_id=None, scheme=None):
        """
        Changes instance according to chosen instance progession.

        Parameters
        ----------
        instance
            Instance specification for potentional new instances
        instance_id
            ID of the instance to switch to
        scheme
            Update scheme for this progression step (either round robin, random or no progression)

        """
        if instance is not None:
            self.instance = instance
        elif instance_id is not None:
            self.inst_id = instance_id
            self.instance = self.instance_set[self.inst_id]
        elif scheme == "round_robin":
            self.instance_index = (self.instance_index + 1) % len(self.instance_id_list)
            self.inst_id = self.instance_id_list[self.instance_index]
            self.instance = self.instance_set[self.inst_id]
        elif scheme == "random":
            self.inst_id = np.random.choice(self.instance_id_list)
            self.instance = self.instance_set[self.inst_id]

    def step(self, action):
        """
        Execute environment step.

        Parameters
        ----------
        action
            Action to take

        Returns
        -------
        state
            Environment state
        reward
            Environment reward
        terminated: bool
            Run finished flag
        truncated: bool
            Run timed out flag
        info : dict
            Additional metainfo

        """
        raise NotImplementedError

    def reset(self, seed: int = None):
        """
        Reset environment.

        Parameters
        ----------
        seed
            Seed for the environment

        Returns
        -------
        state
            Environment state
        info: dict
            Additional metainfo

        """
        raise NotImplementedError

    def get_inst_id(self):
        """
        Return instance ID.

        Returns
        -------
        int
            ID of current instance

        """
        return self.inst_id

    def get_instance_set(self):
        """
        Return instance set.

        Returns
        -------
        list
            List of instances

        """
        return self.instance_set

    def get_instance(self):
        """
        Return current instance.

        Returns
        -------
        type flexible
            Currently used instance

        """
        return self.instance

    def set_inst_id(self, inst_id):
        """
        Change current instance ID.

        Parameters
        ----------
        inst_id : int
            New instance index

        """
        self.inst_id = inst_id
        self.instance_index = self.instance_id_list.index(self.inst_id)

    def set_instance_set(self, inst_set):
        """
        Change instance set.

        Parameters
        ----------
        inst_set: list
            New instance set

        """
        self.instance_set = inst_set
        self.instance_id_list = sorted(list(self.instance_set.keys()))

    def set_instance(self, instance):
        """
        Change currently used instance.

        Parameters
        ----------
        instance:
            New instance

        """
        self.instance = instance

    def seed(self, seed=None, seed_action_space=False):
        """
        Set rng seed.

        Parameters
        ----------
        seed:
            seed for rng
        seed_action_space: bool, default False
            if to seed the action space as well

        """
        self.initial_seed = seed
        # maybe one should use the seed generated by seeding.np_random(seed) but it can be to large see issue https://github.com/openai/gym/issues/2210
        random.seed(seed)
        np.random.seed(seed)
        self.np_random, seed = seeding.np_random(seed)
        # uses the uncorrelated seed from seeding but makes sure that no randomness is introduces.

        if seed_action_space:
            self.action_space.seed(seed)

        return [seed]

    def use_test_set(self):
        """Change to test instance set."""
        if self.test_set is None:
            raise ValueError(
                "No test set was provided, please check your benchmark config."
            )

        self.test = True
        self.training_set = self.instance_set
        self.training_id_list = self.instance_id_list
        self.training_inst_id = self.inst_id
        self.training_instance = self.instance

        self.instance_set = self.test_set
        self.instance_id_list = self.test_instance_id_list
        self.inst_id = self.test_inst_id
        self.instance = self.test_instance

    def use_training_set(self):
        """Change to training instance set."""
        self.test = False
        self.test_set = self.instance_set
        self.test_instance_id_list = self.instance_id_list
        self.test_inst_id = self.inst_id
        self.test_instance = self.instance

        self.instance_set = self.training_set
        self.instance_id_list = self.training_id_list
        self.inst_id = self.training_inst_id
        self.instance = self.training_instance


class OLLGATheoryEnv(AbstractEnv):
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
        super(OLLGATheoryEnv, self).__init__(config)
        self.logger = logging.getLogger(self.__str__())

        self.test_env = test_env

        self.name = config.name

        self.discrete_action = False

        # name of reward function
        assert config.reward_choice in [
            "imp_div_evals",
            "imp_div_evals_new",
            "imp_minus_evals",
            "minus_evals",
            "imp",
            "minus_evals_normalised",
            "imp_minus_evals_normalised",
            "imp_minus_evals_scaling",
            "imp_minus_evals_shifting",
            "imp_minus_evals_scaling_shifting",
            "imp_minus_evals_penalty_shifting",
        ]
        self.reward_choice = config.reward_choice
        # print("Reward choice: " + self.reward_choice)

        # get problem
        self.problem = globals()[config.problem]

        # read names of all observation variables
        self.obs_description = config.observation_description
        self.obs_var_names = [
            s.strip() for s in config.observation_description.split(",")
        ]

        # functions to get values of the current state from histories
        # (see reset() function for those history variables)
        self.state_functions = []
        for var_name in self.obs_var_names:
            if var_name == "n":
                self.state_functions.append(lambda: self.n)
            elif var_name in ["lbd", "mut", "lbd_cross", "cross"]:
                self.state_functions.append(
                    lambda his="history_" + var_name: vars(self)[his][-1]
                )
            elif (
                "_{t-" in var_name
            ):  # TODO: this implementation only allow accessing history of r, but not delta_f(x), optimal_k, etc
                k = int(
                    var_name.split("_{t-")[1][:-1]
                )  # get the number in _{t-<number>}
                name = var_name.split("_{t-")[0]  # get the variable name (r, f(x), etc)
                self.state_functions.append(
                    lambda his="history_" + name: vars(self)[his][-(k + 1)]
                )  # the last element is the value at the current time step, so we have to go one step back to access the history
            elif var_name == "f(x)":
                self.state_functions.append(lambda: self.history_fx[-1])
            elif var_name == "delta_f(x)":
                self.state_functions.append(
                    lambda: self.history_fx[-1] - self.history_fx[-2]
                )
            elif var_name == "optimal_r":
                self.state_functions.append(
                    lambda: int(self.n / (self.history_fx[-1] + 1))
                )
            else:
                raise Exception("Error: invalid state variable name: " + var_name)

        # the random generator used by OLLGA
        if "seed" in config:
            seed = config.seed
        else:
            seed = None
        if "seed" in self.instance:
            seed = self.instance.seed
        self.seed(seed)

        # for logging
        self.outdir = None
        if "outdir" in config:
            self.outdir = config.outdir + "/" + str(uuid.uuid4())

        # setup other variables (that are specific to the current instances)
        self.reset_()

    def get_obs_domain_from_name(var_name):
        """
        Get default lower and upperbound of a observation variable based on its name.

        The observation space will then be created

        Returns
        -------
            Two int values, e.g., -np.inf, np.inf

        """
        return -np.inf, np.inf

    def reset_(self, seed=None, options={}):
        """
        Resets env.

        Returns
        -------
        numpy.array
            Environment state

        """
        # current problem size (n) & evaluation limit (max_evals)
        self.n = self.instance.size
        if self.test_env:
            self.max_evals = self.n_steps
        else:
            self.max_evals = int(0.8 * self.n * self.n)
        # self.logger.info("n:%d, max_evals:%d" % (self.n, self.max_evals))

        # set random seed
        if "seed" in self.instance:
            self.seed(self.instance.seed)

        # create an initial solution
        if self.instance.initObj == "random":
            self.x = self.problem(n=self.instance.size, rng=self.np_random)
        else:
            self.x = self.problem(
                n=self.instance.size, rng=self.np_random, initObj=self.instance.initObj
            )

        # total number of evaluations so far
        self.total_evals = 1

        # reset histories
        self.history_lbd = deque([0] * HISTORY_LENGTH, maxlen=HISTORY_LENGTH)
        self.history_fx = deque(
            [self.x.fitness] * HISTORY_LENGTH, maxlen=HISTORY_LENGTH
        )
        ## dev history of parameters
        self.history_mut = deque([0] * HISTORY_LENGTH, maxlen=HISTORY_LENGTH)
        self.history_lbd_cross = deque([0] * HISTORY_LENGTH, maxlen=HISTORY_LENGTH)
        self.history_cross = deque([0] * HISTORY_LENGTH, maxlen=HISTORY_LENGTH)

        # for debug only
        self.log_r = []
        self.log_reward = []
        self.log_fx = []
        self.init_obj = self.x.fitness

        return self.get_state(), {}

    def reset(self, seed=None, options={}):
        """
        Resets env.

        Returns
        -------
        numpy.array
            Environment state

        """
        super(OLLGATheoryEnv, self).reset_(seed)
        return self.reset_(seed, options)

    def get_state(self):
        """Return state."""
        return np.asarray([f() for f in self.state_functions])

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
        # if isinstance(actions, np.ndarray) or isinstance(actions, list):
        #     assert len(actions) == 1
        #     lbd = actions[0]
        # else:
        lbd = actions

        # if lbd is out of range
        stop = False
        if lbd < 1 or lbd > self.n:
            self.logger.info(f"WARNING: lambda={lbd} is out of bound")

            # if we're in the training phase, we return a large negative reward and stop the episode
            if self.test_env is False:
                terminated = True
                n_evals = 0
                reward = -MAX_INT
                stop = True
            # if we're in the test phase, just clip r back to the range and continue
            else:
                lbd = np.clip(lbd, 1, self.n)

        if stop is False:
            # flip r bits
            lbd = int(lbd)
            mutation_rate = np.float64(lbd / self.n)
            mutation_size = np.int64(lbd)
            crossover_rate = np.float64(1.0 / lbd)
            crossover_size = np.int64(lbd)
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
                reward = (
                    self.x.fitness - fitness_before_update - n_evals + kwargs["shift"]
                )
            elif self.reward_choice == "imp_minus_evals_scaling_shifting":
                reward = (
                    (self.x.fitness - fitness_before_update - n_evals) / self.n
                ) + kwargs["shift"]
            self.log_reward.append(reward)

        # update histories
        self.history_fx.append(self.x.fitness)
        self.history_lbd.append(lbd)

        # update logs
        self.log_r.append(lbd)
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

    def close(self) -> bool:
        """
        Close Env.

        No additional cleanup necessary

        Returns
        -------
        bool
            Closing confirmation

        """
        return True


class OLLGAPPOCombEnv(OLLGATheoryEnv):
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
        super(OLLGAPPOCombEnv, self).__init__(config)

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
        if isinstance(actions, np.ndarray):
            actions = actions.tolist()
        # print(f"State: {fitness_before_update}", "Actions received:", actions)
        if stop is False:
            mutation_size, alpha, crossover_size, gamma = actions
            mutation_rate = alpha * mutation_size / self.n
            crossover_rate = gamma / crossover_size

            ## clip rate to [0,1]
            mutation_rate = np.clip(mutation_rate, 0, 1)
            crossover_rate = np.clip(crossover_rate, 0, 1)
            mutation_size = int(mutation_size)
            crossover_size = int(crossover_size)
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
            if self.x.is_optimal():
                terminated = True
                # print("Optimal solution found!")
            elif self.total_evals >= self.max_evals:
                terminated = True
                print(f"Maximum evaluations reached: {self.total_evals}")
            else:
                terminated = False
            # terminated = (self.total_evals >= self.max_evals) or (self.x.is_optimal())
            # update histories
            self.history_fx.append(self.x.fitness)
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
                ## hard code the shifting bias by problem size
                shift = self.config["fixed_shift"]
                print(f"Applying shift of {shift} for problem size {self.n}")
                reward = (
                    self.x.fitness - fitness_before_update - n_evals + shift
                )

            self.log_reward.append(reward)
        else:
            self.history_fx.append(self.x.fitness)

        self.history_lbd.append(mutation_size)
        self.history_mut.append(mutation_rate)
        self.history_cross.append(crossover_rate)
        self.history_lbd_cross.append(crossover_size)

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


class OLLGAPPOCombEnvDiscrete(OLLGAPPOCombEnv):
    """OLLGA environment where the choices of lambda is discretised."""

    def __init__(self, config, test_env=False):
        """Init env."""
        super(OLLGAPPOCombEnvDiscrete, self).__init__(config, test_env)
        assert (
            "action_choices" in config
        ), "Error: action_choices must be specified in benchmark's config"
        assert isinstance(
            self.action_space, gym.spaces.Discrete
        ), "Error: action space must be discrete"
        assert self.action_space.n == len(config["action_choices"][0][0]) ** 4, (
            "Error: action space's size (%d) must be equal to the len(action_choices) (%d)"
            % (self.action_space.n, len(config["action_choices"][0][0]) ** 4)
        )
        self.discrete_action = True
        action_choices = config["action_choices"][0]
        ## combine the action choices
        self.action_choices = []
        for lbd1 in action_choices[0]:
            for mutation in action_choices[1]:
                for lbd2 in action_choices[2]:
                    for crossover in action_choices[3]:
                        self.action_choices.append([lbd1, mutation, lbd2, crossover])

    def step(self, actions, **kwargs):
        """Take step."""
        action_value = self.action_choices[actions]
        return super(OLLGAPPOCombEnvDiscrete, self).step(action_value, **kwargs)
