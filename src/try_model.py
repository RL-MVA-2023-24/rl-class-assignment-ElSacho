from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
import random
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from evaluate import evaluate_HIV, evaluate_HIV_population
import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
import locale
locale.setlocale(locale.LC_ALL, 'fr_FR')  # Définir la locale en français

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = TimeLimit(
    env=HIVPatient(domain_randomization=True), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.

def greedy_action(network, state):
    device = "cuda" if next(network.parameters()).is_cuda else "cpu"
    with torch.no_grad():
        Q = network(torch.Tensor(state).unsqueeze(0).to(device))
        return torch.argmax(Q).item()

class DQM_model(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, depth):
        super(DQM_model, self).__init__()
        self.input_layer = torch.nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = torch.nn.ModuleList([torch.nn.Linear(hidden_dim, hidden_dim) for _ in range(depth - 1)])
        self.output_layer = torch.nn.Linear(hidden_dim, output_dim)
        self.activation = torch.nn.ReLU()
        
    def forward(self, x):
        x = self.activation(self.input_layer(x))
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        return self.output_layer(x)


# DQN config
config = {'nb_actions': env.action_space.n,
          'learning_rate': 0.001,
          'gamma': 0.99,
          'buffer_size': 30_000,
          'epsilon_min': 0.07,
          'epsilon_max': 1.,
          'epsilon_decay_period': 30_000,
          'epsilon_delay_decay': 4_000,
          'batch_size': 1000,
          'gradient_steps': 3,
          'update_target_strategy': 'replace', # or 'ema'
          'update_target_freq': 400,
          'update_target_tau': 0.005,
          'criterion': torch.nn.MSELoss(),
          'n_state_to_agg': 1
          }

class TestAgent:
    def act(self, observation, use_random=False):
        if use_random:
            return np.random.choice(self.env.action_space.n)
        return greedy_action(self.model, observation)
    
    def save(self, path = "DQN_simple.pt"):
        torch.save({
                    'model_state_dict': self.model.state_dict()
                    }, path)

    def load(self):
        path = "prioritez_replay.pt"
        self.model = DQM_model(6, 256, 4, 6).to(device)
        self.model.load_state_dict(torch.load(path)['model_state_dict'])

agent = TestAgent()
agent.load()
score_agent: float = evaluate_HIV(agent=agent, nb_episode=1)
score_agent_pop: float = evaluate_HIV_population(agent=agent, nb_episode=1)
print("Score default : ",locale.format_string('%d', int(score_agent), grouping=True))
print("Score population : ",locale.format_string('%d', int(score_agent_pop), grouping=True))