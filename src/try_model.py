from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
import random
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from models import DQM_model, RainbowNet, DuelingModel
from tqdm import tqdm
from evaluate import evaluate_HIV, evaluate_HIV_population
import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
import locale
locale.setlocale(locale.LC_ALL, 'fr_FR')  # Définir la locale en français
import sys

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

# FILE_PATH = sys.argv[1]
FILE_PATH = "src/double_and_n.pt"

class TestAgent:
    def act(self, observation, use_random=False):
        if use_random:
            return np.random.choice(self.env.action_space.n)
        return greedy_action(self.model, observation)
    
    def save(self, path = FILE_PATH):
        torch.save({
                    'model_state_dict': self.model.state_dict()
                    }, path)

    def load(self):
        path = FILE_PATH
        config = torch.load(path)['conf']
        v_min = config['v_min']
        config['v_max'] = config['v_max'] * config['n_step_return']
        v_max = config['v_max']
        n_atoms = config['n_atoms']
        support = torch.linspace(v_min, v_max, n_atoms).to(device)
        if config['distributional']:
            self.model = RainbowNet(config['obs_space'], 256, config['nb_actions'], n_atoms, 4, support, config['noisy'], dueling=config['dueling']).to(device)
        elif config['dueling']:
            self.model = DuelingModel(config['obs_space'], 256, config['nb_actions'], 6, noisy=config['noisy']).to(device)
        else:
            self.model = DQM_model(config['obs_space'], 256, config['nb_actions'], 6, noisy=config['noisy']).to(device)
        
        self.model.load_state_dict(torch.load(path)['model_state_dict'])

agent = TestAgent()
agent.load()
score_agent: float = evaluate_HIV(agent=agent, nb_episode=1)
score_agent_pop: float = evaluate_HIV_population(agent=agent, nb_episode=5)
print("Score default : ",locale.format_string('%d', int(score_agent), grouping=True))
print("Score population : ",locale.format_string('%d', int(score_agent_pop), grouping=True))