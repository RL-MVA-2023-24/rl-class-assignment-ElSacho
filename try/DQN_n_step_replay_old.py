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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


class ReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = int(capacity) # capacity of the buffer
        self.data = []
        self.index = 0 # index of the next cell to be filled
        self.device = device
    def append(self, s, a, r, s_, d):
        if len(self.data) < self.capacity:
            self.data.append(None)
        self.data[self.index] = (s, a, r, s_, d)
        self.index = (self.index + 1) % self.capacity
    def sample(self, batch_size):
        batch = random.sample(self.data, batch_size)
        return list(map(lambda x:torch.Tensor(np.array(x)).to(self.device), list(zip(*batch))))
    def __len__(self):
        return len(self.data)
    
def greedy_action(network, state):
    device = "cuda" if next(network.parameters()).is_cuda else "cpu"
    with torch.no_grad():
        Q = network(torch.Tensor(state).unsqueeze(0).to(device))
        return torch.argmax(Q).item()


class dqn_agent:
    def __init__(self, config, model):
        device = "cuda" if next(model.parameters()).is_cuda else "cpu"
        self.nb_actions = config['nb_actions']
        self.gamma = config['gamma'] if 'gamma' in config.keys() else 0.99
        self.batch_size = config['batch_size'] if 'batch_size' in config.keys() else 100
        buffer_size = config['buffer_size'] if 'buffer_size' in config.keys() else int(1e5)
        self.memory = ReplayBuffer(buffer_size,device)
        self.epsilon_max = config['epsilon_max'] if 'epsilon_max' in config.keys() else 1.
        self.epsilon_min = config['epsilon_min'] if 'epsilon_min' in config.keys() else 0.01
        self.epsilon_stop = config['epsilon_decay_period'] if 'epsilon_decay_period' in config.keys() else 1000
        self.epsilon_delay = config['epsilon_delay_decay'] if 'epsilon_delay_decay' in config.keys() else 20
        self.epsilon_step = (self.epsilon_max-self.epsilon_min)/self.epsilon_stop
        self.model = model 
        self.target_model = deepcopy(self.model).to(device)
        self.criterion = config['criterion'] if 'criterion' in config.keys() else torch.nn.MSELoss()
        lr = config['learning_rate'] if 'learning_rate' in config.keys() else 0.001
        self.optimizer = config['optimizer'] if 'optimizer' in config.keys() else torch.optim.Adam(self.model.parameters(), lr=lr)
        self.nb_gradient_steps = config['gradient_steps'] if 'gradient_steps' in config.keys() else 1
        self.update_target_strategy = config['update_target_strategy'] if 'update_target_strategy' in config.keys() else 'replace'
        self.update_target_freq = config['update_target_freq'] if 'update_target_freq' in config.keys() else 20
        self.update_target_tau = config['update_target_tau'] if 'update_target_tau' in config.keys() else 0.005
        self.monitoring_nb_trials = config['monitoring_nb_trials'] if 'monitoring_nb_trials' in config.keys() else 0
        self.monitoring_freq = config['monitoring_freq'] if 'monitoring_freq' in config.keys() else 50

    def MC_eval(self, env, nb_trials):   # NEW NEW NEW
        MC_total_reward = []
        MC_discounted_reward = []
        for _ in tqdm(range(nb_trials)):
            x,_ = env.reset()
            done = False
            trunc = False
            total_reward = 0
            discounted_reward = 0
            step = 0
            while not (done or trunc):
                a = greedy_action(self.model, x)
                y,r,done,trunc,_ = env.step(a)
                x = y
                total_reward += r
                discounted_reward += self.gamma**step * r
                step += 1
            MC_total_reward.append(total_reward)
            MC_discounted_reward.append(discounted_reward)
        return np.mean(MC_discounted_reward), np.mean(MC_total_reward)
    
    def V_initial_state(self, env, nb_trials):   # NEW NEW NEW
        with torch.no_grad():
            for _ in range(nb_trials):
                val = []
                x,_ = env.reset()
                val.append(self.model(torch.Tensor(x).unsqueeze(0).to(device)).max().item())
        return np.mean(val)
    
    def gradient_step(self):
        if len(self.memory) > self.batch_size:
            X, A, R, Y, D = self.memory.sample(self.batch_size)
            QYmax = self.target_model(Y).max(1)[0].detach()
            update = torch.addcmul(R, 1-D, QYmax, value=self.gamma)
            QXA = self.model(X).gather(1, A.to(torch.long).unsqueeze(1))
            loss = self.criterion(QXA, update.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step() 
        
    def train(self, env, max_episode):
        episode_return = []
        MC_avg_total_reward = []   # Monitoring variables
        MC_avg_discounted_reward = []   # Monitoring variables
        V_init_state = []   # Monitoring variables
        episode = 0
        episode_cum_reward = 0
        state, _ = env.reset()
        epsilon = self.epsilon_max
        step = 0
        best_model = 0
        n_step_buffer = []  # Buffer to store n-step transitions
        self.n_step = 5

        while episode < max_episode:
            # update epsilon
            if step > self.epsilon_delay:
                epsilon = max(self.epsilon_min, epsilon - self.epsilon_step)

            # select epsilon-greedy action
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = greedy_action(self.model, state)

            # step
            next_state, reward, done, trunc, _ = env.step(action)
            n_step_buffer.append((state, action, reward, next_state, done))

            if len(n_step_buffer) >= self.n_step or done or trunc:
                R = sum([n_step_buffer[i][2] * (self.gamma ** i) for i in range(len(n_step_buffer))])
                n_state, n_action, _, _, _ = n_step_buffer[0]
                _, _, _, n_next_state, n_done = n_step_buffer[-1]
                self.memory.append(n_state, n_action, R, n_next_state, n_done)
                

            episode_cum_reward += reward

            # train
            for _ in range(self.nb_gradient_steps):
                self.gradient_step()

            # update target network if needed
            if self.update_target_strategy == 'replace':
                if step % self.update_target_freq == 0: 
                    self.target_model.load_state_dict(self.model.state_dict())
            if self.update_target_strategy == 'ema':
                target_state_dict = self.target_model.state_dict()
                model_state_dict = self.model.state_dict()
                tau = self.update_target_tau
                for key in model_state_dict:
                    target_state_dict[key] = tau*model_state_dict + (1-tau)*target_state_dict
                self.target_model.load_state_dict(target_state_dict)

            # next transition
            step += 1
            if done or trunc:
                state, _ = env.reset()
                print("Episode ", '{:2d}'.format(episode), 
                          ", epsilon ", '{:6.2f}'.format(epsilon), 
                          ", batch size ", '{:4d}'.format(len(self.memory)), 
                          ", ep return ", locale.format_string('%d', int(episode_cum_reward), grouping=True),
                          sep='')
                episode_cum_reward = 0
                episode += 1
                n_step_buffer = []  # Reset n-step buffer
            else:
                state = next_state

        return episode_return, MC_avg_discounted_reward, MC_avg_total_reward, V_init_state

    
# Declare network
state_dim = env.observation_space.shape[0]
n_action = env.action_space.n 
nb_neurons= 24

DQN = torch.nn.Sequential(nn.Linear(state_dim, nb_neurons),
                          nn.ReLU(),
                          nn.Linear(nb_neurons, nb_neurons),
                          nn.Tanh(), 
                          nn.Linear(nb_neurons, nb_neurons),
                          nn.ReLU(),
                          nn.Linear(nb_neurons, n_action)).to(device)

class DQM_model(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, depth = 2):
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

model = DQM_model(6, 256, 4, 6).to(device)
# DQN config
config = {'nb_actions': env.action_space.n,
          'learning_rate': 0.001,
          'gamma': 0.99,
          'buffer_size': 1_000_000,
          'epsilon_min': 0.01,
          'epsilon_max': 1.,
          'epsilon_decay_period': 15_000,
          'epsilon_delay_decay': 4_000,
          'batch_size': 1000,
          'gradient_steps': 1,
          'update_target_strategy': 'replace', # or 'ema'
          'update_target_freq': 400,
          'update_target_tau': 0.005,
          'criterion': torch.nn.SmoothL1Loss(),
          'monitoring_nb_trials': 1,
          'monitoring_freq' : 50,
          'save_every': 10}

# Train agent
agent = dqn_agent(config, model)
ep_length, disc_rewards, tot_rewards, V0 = agent.train(env, 2000)
plt.plot(ep_length, label="training episode length")
plt.plot(tot_rewards, label="MC eval of total reward")
plt.legend()
plt.figure()
plt.plot(disc_rewards, label="MC eval of discounted reward")
plt.plot(V0, label="average $max_a Q(s_0)$")
plt.legend();