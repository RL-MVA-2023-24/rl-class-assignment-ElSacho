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

# LIST_PATH = ["gamma_0_95.pt", "lr_7e_5.pt", "lr_5e_5.pt", "lr_5e_3.pt", "lr_5e_4.pt", "prioritized_beta_0_4.pt", "prioritized_beta_0_3.pt", "prioritized_beta_0_5.pt", "prioritized_beta_0_6.pt"]
LIST_PATH = ["gamma_0_95.pt", "lr_7e_5.pt", "lr_5e_5.pt", "lr_5e_3.pt", "lr_5e_4.pt", "prioritized_beta_0_4.pt", "prioritized_beta_0_3.pt", "prioritized_beta_0_5.pt", "prioritized_beta_0_6.pt"]
# LIST_PATH = ["gamma_0_95.pt", "lr_7e_5.pt",  "prioritized_beta_0_4.pt", "dueling_and_n.pt", "double_and_n.pt"]

class TestAgent:
    def act_vote(self, observation, use_random=False):
        if use_random:
            return np.random.choice(self.env.action_space.n)
        action_votes = [0,0,0,0]
        for model in self.models:
            action = greedy_action(model, observation)
            action_votes[action] += 1
        action_votes = np.array(action_votes)
        return np.argmax(action_votes)
    
    def act_score(self, observation, use_random=False):
        if use_random:
            return np.random.choice(self.env.action_space.n)
        action_score = torch.zeros(4).unsqueeze(0).to(device)
        for model in self.models:
            action_score += model(torch.Tensor(observation).unsqueeze(0).to(device))
        return torch.argmax(action_score).item()
    
    def act_best_score(self, observation, use_random=False):
        action = 0
        best_action_score = 0
        if use_random:
            return np.random.choice(self.env.action_space.n)
        for model in self.models:
            action_score = model(torch.Tensor(observation).unsqueeze(0).to(device))
            max_val = torch.max(action_score).item()
            if max_val > best_action_score:
                action = torch.argmax(action_score).item()
        return action
        
    
    def act(self, observation, use_random=False, method = "score"):
        return self.act_best_score(observation, use_random)
        return self.act_score(observation, use_random) if method == "score" else self.act_vote(observation, use_random) 
    
    def save(self, path = None):
        return 
        torch.save({
                    'model_state_dict': self.model.state_dict()
                    }, path)
        
    def load(self):
        self.models = []  
        for path in LIST_PATH:
            self.models.append(self.load_one_model(path))

    def load_one_model(self, path):
        config = torch.load(path)['conf']
        v_min = config['v_min']
        config['v_max'] = config['v_max'] * config['n_step_return']
        v_max = config['v_max']
        n_atoms = config['n_atoms']
        support = torch.linspace(v_min, v_max, n_atoms).to(device)
        if config['distributional']:
            model = RainbowNet(config['obs_space'], 256, config['nb_actions'], n_atoms, 4, support, config['noisy'], dueling=config['dueling']).to(device)
        elif config['dueling']:
            model = DuelingModel(config['obs_space'], 256, config['nb_actions'], 6, noisy=config['noisy']).to(device)
        else:
            model = DQM_model(config['obs_space'], 256, config['nb_actions'], 6, noisy=config['noisy']).to(device)
        model.load_state_dict(torch.load(path)['model_state_dict'])
        return model

agent = TestAgent()
agent.load()
score_agent: float = evaluate_HIV(agent=agent, nb_episode=1)
print("Score default : ",locale.format_string('%d', int(score_agent), grouping=True))
score_agent_pop: float = evaluate_HIV_population(agent=agent, nb_episode=5)
print("Score population : ",locale.format_string('%d', int(score_agent_pop), grouping=True))