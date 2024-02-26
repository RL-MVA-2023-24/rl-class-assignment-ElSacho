from copy import deepcopy

import numpy as np
from gym.spaces import Discrete, Dict, Box
from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient


class CartPole:
    """
    Wrapper for gym CartPole environment
    """

    def __init__(self, config=None):
        self.env = TimeLimit(
            env=HIVPatient(domain_randomization=False), max_episode_steps=200
        )
        self.action_space = Discrete(2)
        self.observation_space = self.env.observation_space

    def reset(self):
        return self.env.reset()[0]

    def step(self, action):
        obs, rew, done, trunc, _ = self.env.step(action)
        return obs, rew, trunc, trunc

    def set_state(self, state):
        self.env = deepcopy(state)
        # obs = np.array(list(self.env.unwrapped.state))
        return None

    def get_state(self):
        return deepcopy(self.env)

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()


