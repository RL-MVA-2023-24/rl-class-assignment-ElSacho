from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
import random
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from evaluate import evaluate_HIV
import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
import locale
locale.setlocale(locale.LC_ALL, 'fr_FR')  # Définir la locale en français



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

class tab_state:
    def __init__(self, obs_size, n_state_to_agg):
        self.obs_size = obs_size
        self.n_state_to_agg = n_state_to_agg
        self.tab_state = np.zeros(self.obs_size * self.n_state_to_agg)
        
    def push(self, state):
        # Shift the existing state data to the left by one state's worth of data
        for i in range(self.n_state_to_agg - 1):
            self.tab_state[i * self.obs_size: (i + 1) * self.obs_size] = self.tab_state[(i + 1) * self.obs_size: (i + 2) * self.obs_size]
        
        # Insert the new state at the end
        self.tab_state[(self.n_state_to_agg - 1) * self.obs_size: self.n_state_to_agg * self.obs_size] = state

        
    def reset(self):
        self.tab_state = np.zeros(self.obs_size * self.n_state_to_agg)
        


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
        self.obs_space = config['obs_space'] if 'obs_space' in config.keys() else 6
        self.target_model = deepcopy(self.model).to(device)
        self.criterion = config['criterion'] if 'criterion' in config.keys() else torch.nn.MSELoss()
        lr = config['learning_rate'] if 'learning_rate' in config.keys() else 0.001
        self.optimizer = config['optimizer'] if 'optimizer' in config.keys() else torch.optim.Adam(self.model.parameters(), lr=lr)
        self.nb_gradient_steps = config['gradient_steps'] if 'gradient_steps' in config.keys() else 1
        self.update_target_strategy = config['update_target_strategy'] if 'update_target_strategy' in config.keys() else 'replace'
        self.update_target_freq = config['update_target_freq'] if 'update_target_freq' in config.keys() else 20
        self.update_target_tau = config['update_target_tau'] if 'update_target_tau' in config.keys() else 0.005
        self.n_state_to_agg = config['n_state_to_agg'] if 'n_state_to_agg' in config.keys() else 1
        

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
        episode = 0
        episode_cum_reward = 0
        obs_space = self.obs_space
        best_score = 0
        states = tab_state(obs_space, self.n_state_to_agg)
        step_ep = 0
        state, _ = env.reset()
        last_action = 0
        state_a = np.zeros(4+6)
        state_a[:6] = state
        states.push(state)
        epsilon = self.epsilon_max
        step = 0
        best_model = 0
        while episode < max_episode:
            # update epsilon
            if step > self.epsilon_delay:
                epsilon = max(self.epsilon_min, epsilon-self.epsilon_step)
            # select epsilon-greedy action
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                # action = greedy_action(self.model, states.tab_state)
                if self.n_state_to_agg == 1:
                    action = greedy_action(self.model, state)
                else:
                    state_a[5 + last_action] = 1
                    state_a[:6] = state
                    action = greedy_action(self.model, state_a)
                    # action = greedy_action(self.model, states.tab_state)
            # step
            next_state, reward, done, trunc, _ = env.step(action)
            # reward *= 1e-6
            last_action = action
            next_state_a = np.zeros(4+6)
            next_state_a[5 + last_action] = 1
            next_state_a[:6] = next_state
            # last_states = states.tab_state.copy()
            # state *= 1e-3
            states.push(next_state)
            step_ep += 1
            step_ep = step_ep % self.n_state_to_agg
            # self.memory.append(last_states, action, reward, states.tab_state, done)
            if self.n_state_to_agg == 1:
                self.memory.append(state, action, reward, next_state, done)
            else:
                self.memory.append(next_state_a, action, reward, state_a, done)
                # self.memory.append(last_states, action, reward, states.tab_state, done)
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
                episode += 1
                # Monitoring
                episode_return.append(episode_cum_reward)
                print("Episode ", '{:2d}'.format(episode), 
                        ", epsilon ", '{:6.2f}'.format(epsilon), 
                        ", batch size ", '{:4d}'.format(len(self.memory)), 
                        ", ep return ", locale.format_string('%d', int(episode_cum_reward), grouping=True),
                        sep='')
                
                state, _ = env.reset()
                last_action = 0
                state_a[:6] = state
                states.reset()
                # state *= 1e-3
                states.push(state)
                episode_cum_reward = 0
                try:
                    score_agent: float = evaluate_HIV(agent=self, nb_episode=1)
                    print(locale.format_string('%d', int(score_agent), grouping=True))
                    if score_agent > best_score:
                        agent.save()
                        best_score = score_agent
                except:
                    print("could not test the model")
            else:
                state = next_state
            
        return episode_return
    
    def act(self, observation, use_random=False):
        if use_random:
            return np.random.choice(self.env.action_space.n)
        return greedy_action(self.model, observation)
    
    def save(self, path = "DQN_simple.pt"):
        torch.save({
                    'model_state_dict': self.model.state_dict()
                    }, path)

    def load(self):
        path = "DQN_simple.pt"
        self.model = DQM_model(6*config['n_state_to_agg'], 256, 4, 6).to(device)
        self.model.load_state_dict(torch.load(path)['model_state_dict'])
    

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.

# import gymnasium as gym
# env = gym.make('CartPole-v1', render_mode="rgb_array")
# import torch
# import torch.nn as nn
# import matplotlib.pyplot as plt
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Declare network
state_dim = env.observation_space.shape[0]
n_action = env.action_space.n 
nb_neurons= 24
# DQN config
config = {'nb_actions': env.action_space.n,
          'learning_rate': 0.001,
          'gamma': 0.99,
          'buffer_size': 40_000,
          'epsilon_min': 0.01,
          'epsilon_max': 1.,
          'epsilon_decay_period': 15_000,
          'epsilon_delay_decay': 4_000,
          'batch_size': 500,
          'gradient_steps': 3,
          'update_target_strategy': 'replace', # or 'ema'
          'update_target_freq': 400,
          'update_target_tau': 0.005,
          'obs_space' : state_dim,
          'criterion': torch.nn.MSELoss(),
          'n_state_to_agg': 2
          }

# config = {'nb_actions': env.action_space.n,
#           'learning_rate': 0.001,
#           'gamma': 0.99,
#           'buffer_size': 20_000,
#           'epsilon_min': 0.01,
#           'epsilon_max': 1.,
#           'epsilon_decay_period': 2_000,
#           'epsilon_delay_decay': 300,
#           'batch_size': 10,
#           'gradient_steps': 3,
#           'update_target_strategy': 'replace', # or 'ema'
#           'update_target_freq': 400,
#           'update_target_tau': 0.005,
#           'obs_space' : state_dim,
#           'criterion': torch.nn.MSELoss(),
#           'n_state_to_agg': 1
#           }


model = DQM_model(10, 256, n_action, 6).to(device)

# Train agent
agent = dqn_agent(config, model)
scores = agent.train(env, 200)
plt.plot(scores)
