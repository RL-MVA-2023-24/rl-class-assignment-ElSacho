from copy import deepcopy

import gym
import numpy as np
from gym.spaces import Discrete, Dict, Box


class CartPole:
    """
    Wrapper for gym CartPole environment
    """

    def __init__(self, config=None):
        self.env = gym.make("CartPole-v0")
        self.action_space = Discrete(2)
        self.observation_space = self.env.observation_space

    def reset(self):
        return self.env.reset()[0]

    def step(self, action):
        obs, rew, done, trunc, _ = self.env.step(action)
        return obs, rew, done, trunc

    def set_state(self, state):
        self.env = deepcopy(state)
        obs = np.array(list(self.env.unwrapped.state))
        return obs

    def get_state(self):
        return deepcopy(self.env)

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()


