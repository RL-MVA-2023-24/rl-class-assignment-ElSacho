from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
import random
import torch
import numpy as np
import torch.nn as nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def greedy_action(network, state):
    device = "cuda" if next(network.parameters()).is_cuda else "cpu"
    with torch.no_grad():
        Q = network(torch.Tensor(state).unsqueeze(0).to(device))
        return torch.argmax(Q).item()
    
class ProjectAgent:    
    def act(self, observation, use_random=False):
        if use_random:
            return np.random.choice(4)
        else:
            return greedy_action(self.model, observation)
        

    def save(self, path):
        torch.save({
                    'model_state_dict': self.model.state_dict(),
                    }, path)

    def load(self):
        checkpoint = torch.load("src/best_agent_path.pt")
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        