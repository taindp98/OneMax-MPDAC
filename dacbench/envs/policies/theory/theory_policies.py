import os
import sys
from typing import List

from dacbench.abstract_agent import AbstractDACBenchAgent

sys.path.append(os.path.dirname(__file__))
# import pprint

from dacbench.envs.policies.theory.calculate_optimal_policy.run import (
    calculate_optimal_policy,
)


class TheoryOptimalAgent(AbstractDACBenchAgent):
    def __init__(self, env, name="TheoryOptimal"):
        self.name = name
        self.env = env

    def act(self, state, reward) -> int:
        """
        Return the best radius for the current objective value
        """
        k = int(self.env.n / (self.env.x.fitness + 1))
        return k

    def get_actions_for_all_states(self, n) -> List[int]:
        """
        Return the best radii for all objective function values from 0 to n-1
        """
        return [int(n / (fx + 1)) for fx in range(0, n)]

    def train(self, state, reward):
        pass

    def end_episode(self, state, reward):
        pass


class TheoryOptimalDiscreteAgent(AbstractDACBenchAgent):
    def __init__(self, env, name="TheoryOptimalDiscrete"):
        self.name = name
        self.env = env
        assert "action_choices" in env.unwrapped.__dict__
        # env.reset()

        # read all calculated optimal policies
        script_dir = os.path.dirname(__file__)
        positions_dict = self.parse_rls_optimal_policies(
            fn=os.path.join(script_dir, "optimal_policies.txt")
        )

        self.positions = {}
        for inst_id, action_choices in env.action_choices.items():
            acts = env.action_choices[inst_id]
            acts_str = str(acts)  # string of all actions
            inst = env.instance_set[inst_id]
            n = inst["size"]
            sn = str(n)
            # TODO: calculate optimal policy if it doesn't exist in optimal_policies.txt using Martin's D code
            assert (
                sn in positions_dict
            ), "ERROR: the optimal policy for this setting hasn't been calculated yet. Please see documentation on how to calculate the optimal policy for a new setting."
            assert (
                acts_str in positions_dict[sn]
            ), "ERROR: the optimial policy for this setting hasn't been calculated yet. Please see documentation on how to calculate the optimal policy for a new setting."
            # self.positions[inst_id] = positions_dict[sn][acts_str]
            self.positions[n] = positions_dict[sn][acts_str]

    def act(self, state, reward) -> int:
        """
        Return index of the best radius for the current objective value
        """
        fx = self.env.x.fitness
        act = None
        positions = self.positions[self.env.n]
        for i in range(len(positions) - 1, -1, -1):
            if positions[i] >= fx:
                act = i
                break
        # print(f"{fx:3d}: {self.acts[act]:3d}")
        assert act is not None, f"ERROR: no optimal action found for f(x)={fx}."
        return act

    def parse_rls_optimal_policies(self, fn="optimal_policies.txt"):
        with open(fn, "rt") as f:
            ls = [s.replace("\n", "") for s in f.readlines()]
            ls = [s for s in ls if s != ""]
        positions_dict = {}
        for s in ls:
            ls1 = s.split(";")
            n = int(ls1[0])
            ks = [x.strip() for x in ls1[1].split(",") if x.strip() != ""]
            pos = [x.strip() for x in ls1[2].split(",") if x.strip() != ""]
            assert len(ks) == len(pos), f"ERROR with {s} ({len(ks)} {len(pos)})"
            ks = "[" + ", ".join([str(x) for x in ks]) + "]"
            pos = [int(x) for x in pos]
            sn = str(n)
            if sn not in positions_dict:
                positions_dict[sn] = {}
            positions_dict[sn][ks] = pos

        # pprint.pprint(positions_dict)
        return positions_dict

    def get_actions_for_all_states(self, n) -> List[int]:
        """
        Return the best radii for all objective function values from 0 to n-1
        """
        positions = self.positions[n]

        def get_optimal_act(fx):
            act = None
            for i in range(len(positions) - 1, -1, -1):
                if positions[i] >= fx:
                    act = i
                    break
            assert act is not None, f"ERROR: invalid f(x) ({fx})"
            return act

        # action_choices = self.env.action_choices[self.env.inst_id]
        return [get_optimal_act(fx) for fx in range(0, n)]
        return

    def train(self, state, reward):
        pass

    def end_episode(self, state, reward):
        pass
