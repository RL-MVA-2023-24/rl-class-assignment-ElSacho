from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
import random
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from models import DQM_model, RainbowNet, DuelingModel

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
    
class Model(nn.Module):
    def __init__(self, hidden_size):
        super(Model, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hidden = nn.Linear(6, hidden_size).to(self.device)
        self.hidden2 = nn.Linear(hidden_size, hidden_size).to(self.device)
        self.output = nn.Linear(hidden_size, 4).to(self.device)
        
    def forward(self, x):
        # x = torch.from_numpy(x).to(self.device)
        x = x.to(next(self.parameters()).dtype)
        x = F.relu(self.hidden(x))
        x = F.relu(self.hidden2(x))
        x = self.output(x)
        x = x.unsqueeze(0)
        return F.softmax(x, dim=1)
    
class ProjectAgent:  
    def act(self, observation, use_random=False):
        if use_random:
            a = np.random.choice(4)
            # print(a)
            return a
        else:
            a = greedy_action(self.model, observation)
            # print(a)
            return a
        
    def save(self, path):
        print("saving")
        torch.save({
                    'model_state_dict': self.model.state_dict(),
                    }, path)

    def load(self):
        path = "src/double_and_n.pt"
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
        self.model.eval()
        
        # etat_du_modele = torch.load("src/DQN_simple.pt", map_location=torch.device('cpu'))
        # self.model.load_state_dict(etat_du_modele)
        # self.model.eval()