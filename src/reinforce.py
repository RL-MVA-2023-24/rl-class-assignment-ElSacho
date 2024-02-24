import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class policyNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        state_dim = env.observation_space.shape[0]
        n_action = env.action_space.n
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, n_action)

    def forward(self, x):
        x = x.float()
        if x.dim() == 1:
            x = x.unsqueeze(dim=0)
        x = F.relu(self.fc1(x))
        action_scores = self.fc2(x)
        return F.softmax(action_scores,dim=1)

    def sample_action(self, x):
        probabilities = self.forward(x)
        action_distribution = Categorical(probabilities)
        return action_distribution.sample().item()

    def log_prob(self, x, a):
        probabilities = self.forward(x)
        action_distribution = Categorical(probabilities)
        return action_distribution.log_prob(a)
    
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange

class reinforce_agent:
    def __init__(self, config, policy_network):
        self.device = "cuda" if next(policy_network.parameters()).is_cuda else "cpu"
        self.scalar_dtype = next(policy_network.parameters()).dtype
        self.policy = policy_network
        self.gamma = config['gamma'] if 'gamma' in config.keys() else 0.99
        lr = config['learning_rate'] if 'learning_rate' in config.keys() else 0.001
        self.optimizer = torch.optim.Adam(list(self.policy.parameters()),lr=lr)
        self.nb_episodes = config['nb_episodes'] if 'nb_episodes' in config.keys() else 1
    
    def one_gradient_step(self, env):
        # run trajectories until done
        episodes_sum_of_rewards = []
        states = []
        actions = []
        returns = []
        for ep in range(self.nb_episodes):
            x,_ = env.reset()
            rewards = []
            episode_cum_reward = 0
            while(True):
                a = self.policy.sample_action(torch.as_tensor(x))
                y,r,done,trunc,_ = env.step(a)
                states.append(x)
                actions.append(a)
                rewards.append(r)
                episode_cum_reward += r
                x=y
                if done or trunc: 
                    # The condition above should actually be "done or trunc" so that we 
                    # terminate the rollout also if trunc=True.
                    # But then, our return-to-go computation would be biased as we would 
                    # implicitly assume no rewards can be obtained after truncation, which 
                    # is wrong.
                    # We leave it as is for now (which means we will call .step() even 
                    # after trunc=True) and will discuss it later.
                    # Compute returns-to-go
                    new_returns = []
                    G_t = 0
                    for r in reversed(rewards):
                        G_t = r + self.gamma * G_t
                        new_returns.append(G_t)
                    new_returns = list(reversed(new_returns))
                    returns.extend(new_returns)
                    episodes_sum_of_rewards.append(episode_cum_reward)
                    print( locale.format_string('%d', int(episode_cum_reward), grouping=True))
                    break
        # make loss
        returns = torch.tensor(returns)
        log_prob = self.policy.log_prob(torch.as_tensor(np.array(states)),torch.as_tensor(np.array(actions)))
        loss = -(returns * log_prob).mean()
        # gradient step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return np.mean(episodes_sum_of_rewards)

    def train(self, env, nb_rollouts):
        avg_sum_rewards = []
        for ep in trange(nb_rollouts):
            avg_sum_rewards.append(self.one_gradient_step(env))
        return avg_sum_rewards
    
import gymnasium as gym
import matplotlib.pyplot as plt

from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
import random
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
import locale
locale.setlocale(locale.LC_ALL, 'fr_FR')  # Définir la locale en français

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.



config = {'gamma': .99,
          'learning_rate': 0.01,
          'nb_episodes': 1
         }

pi = policyNetwork(env)
agent = reinforce_agent(config, pi)
returns = agent.train(env,50)
plt.savefig(returns, "reinforce.png")