from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
import torch
import numpy as np
import torch.nn as nn
from evaluate import evaluate_HIV, evaluate_HIV_population
import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
import locale
import torch.nn.functional as F
locale.setlocale(locale.LC_ALL, 'fr_FR')  # Définir la locale en français

import math

class RainbowNet(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, atom_size, depth, support, noisy = False, std_init = 0.5, dueling = True):
        super(RainbowNet, self).__init__()

        self.support = support
        self.out_dim = output_dim
        self.atom_size = atom_size
        
        self.activation = torch.nn.ReLU()
        
        self.input_layer = torch.nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = torch.nn.ModuleList([torch.nn.Linear(hidden_dim, hidden_dim) for _ in range(depth - 1)])
        
        self.dueling = dueling
        if dueling:
            if noisy:
                self.advantage_hidden_layer = NoisyLinear(hidden_dim, hidden_dim, std_init)
                self.advantage_layer = NoisyLinear(hidden_dim, output_dim * atom_size, std_init)
                self.value_hidden_layer = NoisyLinear(hidden_dim, hidden_dim,  std_init)
                self.value_layer = NoisyLinear(hidden_dim, atom_size, std_init)
            else:
                self.advantage_hidden_layer = torch.nn.Linear(hidden_dim, hidden_dim)
                self.advantage_layer = torch.nn.Linear(hidden_dim, output_dim * atom_size)
                self.value_hidden_layer = torch.nn.Linear(hidden_dim, hidden_dim)
                self.value_layer = torch.nn.Linear(hidden_dim, atom_size)
        else:
            if noisy:
                self.output_layer_hidden = NoisyLinear(hidden_dim, hidden_dim, std_init)
                self.output_layer = NoisyLinear(hidden_dim, output_dim, std_init)
            else:
                self.output_layer_hidden = torch.nn.Linear(hidden_dim, hidden_dim * atom_size)
                self.output_layer = torch.nn.Linear(hidden_dim, output_dim * atom_size)
                
    def forward(self, x):
        dist = self.dist(x)
        q = torch.sum(dist * self.support, dim=2)
        return q
    
    def dist(self, x):
        x = self.activation(self.input_layer(x))
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        if self.dueling:
            adv_hid = F.relu(self.advantage_hidden_layer(x))
            val_hid = F.relu(self.value_hidden_layer(x))
        
            advantage = self.advantage_layer(adv_hid).view(-1, self.out_dim, self.atom_size)
            value = self.value_layer(val_hid).view(-1, 1, self.atom_size)
            q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)
        else:
            self.output_layer_hidden(x)
            q_atoms = self.output_layer(x)
        dist = F.softmax(q_atoms, dim=-1)
        dist = dist.clamp(min=1e-3)  # for avoiding nans
        
        return dist
    
    def reset_noise(self):
        """Reset all noisy layers."""
        if self.dueling:
            self.advantage_hidden_layer.reset_noise()
            self.advantage_layer.reset_noise()
            self.value_hidden_layer.reset_noise()
            self.value_layer.reset_noise()
        else:
            self.output_layer_hidden.reset_noise()
            self.output_layer.reset_noise()

class DistribNet(nn.Module):
    def __init__(
        self, 
        in_dim: int, 
        out_dim: int, 
        atom_size: int, 
        support: torch.Tensor
    ):
        """Initialization."""
        super(DistribNet, self).__init__()

        self.support = support
        self.out_dim = out_dim
        self.atom_size = atom_size
        
        self.layers = nn.Sequential(
            nn.Linear(in_dim, 128), 
            nn.ReLU(),
            nn.Linear(128, 128), 
            nn.ReLU(), 
            nn.Linear(128, out_dim * atom_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        dist = self.dist(x)
        q = torch.sum(dist * self.support, dim=2)
        
        return q
    
    def dist(self, x: torch.Tensor) -> torch.Tensor:
        """Get distribution for atoms."""
        q_atoms = self.layers(x).view(-1, self.out_dim, self.atom_size)
        dist = F.softmax(q_atoms, dim=-1)
        dist = dist.clamp(min=1e-3)  # for avoiding nans
        
        return dist

class DQM_model(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, depth, noisy = False, std_init = 0.5):
        super(DQM_model, self).__init__()
        if noisy:
            self.input_layer = NoisyLinear(input_dim, hidden_dim, std_init)
            self.hidden_layers = torch.nn.ModuleList([NoisyLinear(hidden_dim, hidden_dim, std_init) for _ in range(depth - 1)])
            self.output_layer = NoisyLinear(hidden_dim, output_dim, std_init)
        else:
            self.input_layer = torch.nn.Linear(input_dim, hidden_dim)
            self.hidden_layers = torch.nn.ModuleList([torch.nn.Linear(hidden_dim, hidden_dim) for _ in range(depth - 1)])
            self.output_layer = torch.nn.Linear(hidden_dim, output_dim)
        self.activation = torch.nn.ReLU()
        
    def forward(self, x):
        x = self.activation(self.input_layer(x))
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        return self.output_layer(x)
    
    def reset_noise(self):
        """Reset all noisy layers."""
        for layer in self.hidden_layers:
            layer.reset_noise()
        self.input_layer.reset_noise()
        self.output_layer.reset_noise()
    
class DuelingModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, depth, noisy = False, std_init = 0.5):
        super(DuelingModel, self).__init__()
        if noisy:
            self.input_layer = NoisyLinear(input_dim, hidden_dim, std_init)
            self.hidden_layers = torch.nn.ModuleList([NoisyLinear(hidden_dim, hidden_dim, std_init) for _ in range(depth - 1)])
            self.value = NoisyLinear(hidden_dim, 1, std_init)
            self.advantage = NoisyLinear(hidden_dim, output_dim, std_init)
        else:
            self.input_layer = torch.nn.Linear(input_dim, hidden_dim)
            self.hidden_layers = torch.nn.ModuleList([torch.nn.Linear(hidden_dim, hidden_dim) for _ in range(depth - 1)])
            self.value = torch.nn.Linear(hidden_dim, 1)
            self.advantage = torch.nn.Linear(hidden_dim, output_dim)
        self.activation = torch.nn.ReLU()
        
    def forward(self, x):
        x = self.activation(self.input_layer(x))
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        adv = self.advantage(x)
        q_val = self.value(x) + adv - adv.mean(dim=-1, keepdim=True)
        return q_val
    
    def reset_noise(self):
        """Reset all noisy layers."""
        for layer in self.hidden_layers:
            layer.reset_noise()
        self.value.reset_noise()
        self.advantage.reset_noise()
       
class NoisyLinear(nn.Module):
    """Noisy linear module for NoisyNet.
    
    Attributes:
        in_features (int): input size of linear module
        out_features (int): output size of linear module
        std_init (float): initial std value
        weight_mu (nn.Parameter): mean value weight parameter
        weight_sigma (nn.Parameter): std value weight parameter
        bias_mu (nn.Parameter): mean value bias parameter
        bias_sigma (nn.Parameter): std value bias parameter
        
    """

    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        """Initialization."""
        super(NoisyLinear, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(
            torch.Tensor(out_features, in_features)
        )
        self.register_buffer(
            "weight_epsilon", torch.Tensor(out_features, in_features)
        )

        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        self.register_buffer("bias_epsilon", torch.Tensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        """Reset trainable network parameters (factorized gaussian noise)."""
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(0, mu_range)
        self.weight_sigma.data.fill_(
            self.std_init / math.sqrt(self.in_features)
        )
        self.bias_mu.data.uniform_(0, mu_range)
        self.bias_sigma.data.fill_(
            self.std_init / math.sqrt(self.out_features)
        )

    def reset_noise(self):
        """Make new noise."""
        epsilon_in = self.scale_noise(self.in_features)
        epsilon_out = self.scale_noise(self.out_features)

        # outer product
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation.
        
        We don't use separate statements on train / eval mode.
        It doesn't show remarkable difference of performance.
        """
        return F.linear(
            x,
            self.weight_mu + self.weight_sigma * self.weight_epsilon,
            self.bias_mu + self.bias_sigma * self.bias_epsilon,
        )
    
    @staticmethod
    def scale_noise(size: int) -> torch.Tensor:
        """Set scale to make noise (factorized gaussian noise)."""
        x = torch.randn(size)

        return x.sign().mul(x.abs().sqrt())
    
class Network(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, std_init: float = 0.5):
        """Initialization."""
        super(Network, self).__init__()

        self.feature = nn.Linear(in_dim, 128, std_init)
        self.noisy_layer1 = NoisyLinear(128, 128, std_init)
        # self.noisy_layer2 = NoisyLinear(128, 256, std_init)
        # self.noisy_layer3 = NoisyLinear(256, 256, std_init)
        # self.noisy_layer4 = NoisyLinear(256, 128, std_init)
        self.noisy_layer5 = NoisyLinear(128, out_dim, std_init)

    def forward(self, x):
        """Forward method implementation."""
        feature = F.relu(self.feature(x))
        hidden = F.relu(self.noisy_layer1(feature))
        # hidden = F.relu(self.noisy_layer2(hidden))
        # hidden = F.relu(self.noisy_layer3(hidden))
        # hidden = F.relu(self.noisy_layer4(hidden))
        out = self.noisy_layer5(hidden)
        
        return out
    
    def reset_noise(self):
        """Reset all noisy layers."""
        self.noisy_layer1.reset_noise()
        # self.noisy_layer2.reset_noise()
        # self.noisy_layer3.reset_noise()
        # self.noisy_layer4.reset_noise()
        self.noisy_layer5.reset_noise()