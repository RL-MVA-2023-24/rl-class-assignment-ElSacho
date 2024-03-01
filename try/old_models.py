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
import torch.nn.functional as F
locale.setlocale(locale.LC_ALL, 'fr_FR')  # Définir la locale en français

import math

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
    
class DuelingModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, depth):
        super(DuelingModel, self).__init__()
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
      
# import math
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.autograd import Variable

# import numpy as np


# class NoisyLinear(nn.Linear):
#     def __init__(self, in_features, out_features, sigma_init=0.017, bias=True):
#         super(NoisyLinear, self).__init__(in_features, out_features, bias=bias)
#         self.sigma_weight = nn.Parameter(torch.Tensor(out_features, in_features).fill_(sigma_init))
#         self.register_buffer("epsilon_weight", torch.zeros(out_features, in_features))
#         if bias:
#             self.sigma_bias = nn.Parameter(torch.Tensor(out_features).fill_(sigma_init))
#             self.register_buffer("epsilon_bias", torch.zeros(out_features))
#         self.reset_parameters()

#     def reset_parameters(self):
#         std = math.sqrt(3 / self.in_features)
#         nn.init.uniform(self.weight, -std, std)
#         nn.init.uniform(self.bias, -std, std)

#     def forward(self, input):
#         torch.randn(self.epsilon_weight.size(), out=self.epsilon_weight)
#         bias = self.bias
#         if bias is not None:
#             torch.randn(self.epsilon_bias.size(), out=self.epsilon_bias)
#             bias = bias + self.sigma_bias * Variable(self.epsilon_bias)
#         return F.linear(input, self.weight + self.sigma_weight * Variable(self.epsilon_weight), bias)


# class NoisyFactorizedLinear(nn.Linear):
#     """
#     NoisyNet layer with factorized gaussian noise

#     N.B. nn.Linear already initializes weight and bias to
#     """
#     def __init__(self, in_features, out_features, sigma_zero=0.4, bias=True):
#         super(NoisyFactorizedLinear, self).__init__(in_features, out_features, bias=bias)
#         sigma_init = sigma_zero / math.sqrt(in_features)
#         self.sigma_weight = nn.Parameter(torch.Tensor(out_features, in_features).fill_(sigma_init))
#         self.register_buffer("epsilon_input", torch.zeros(1, in_features))
#         self.register_buffer("epsilon_output", torch.zeros(out_features, 1))
#         if bias:
#             self.sigma_bias = nn.Parameter(torch.Tensor(out_features).fill_(sigma_init))

#     def forward(self, input):
#         torch.randn(self.epsilon_input.size(), out=self.epsilon_input)
#         torch.randn(self.epsilon_output.size(), out=self.epsilon_output)

#         func = lambda x: torch.sign(x) * torch.sqrt(torch.abs(x))
#         eps_in = func(self.epsilon_input)
#         eps_out = func(self.epsilon_output)

#         bias = self.bias
#         if bias is not None:
#             bias = bias + self.sigma_bias * Variable(eps_out.t())
#         noise_v = Variable(torch.mul(eps_in, eps_out))
#         return F.linear(input, self.weight + self.sigma_weight * noise_v, bias)


# class DQN(nn.Module):
#     def __init__(self, input_shape, n_actions):
#         super(DQN, self).__init__()

#         self.conv = nn.Sequential(
#             nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, kernel_size=4, stride=2),
#             nn.ReLU(),
#             nn.Conv2d(64, 64, kernel_size=3, stride=1),
#             nn.ReLU()
#         )

#         conv_out_size = self._get_conv_out(input_shape)
#         self.fc = nn.Sequential(
#             nn.Linear(conv_out_size, 512),
#             nn.ReLU(),
#             nn.Linear(512, n_actions)
#         )

#     def _get_conv_out(self, shape):
#         o = self.conv(Variable(torch.zeros(1, *shape)))
#         return int(np.prod(o.size()))

#     def forward(self, x):
#         fx = x.float() / 256
#         conv_out = self.conv(fx).view(fx.size()[0], -1)
#         return self.fc(conv_out)
     
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