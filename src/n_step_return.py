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
from collections import deque
locale.setlocale(locale.LC_ALL, 'fr_FR')  # Définir la locale en français

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


class PrioritizedStandardBuffer():
	def __init__(self, state_dim, batch_size, buffer_size, device, prioritized):
		self.batch_size = batch_size
		self.max_size = int(buffer_size)
		self.device = device

		self.ptr = 0
		self.size = 0

		self.state = np.zeros((self.max_size, state_dim))
		self.action = np.zeros((self.max_size, 1))
		self.next_state = np.array(self.state)
		self.reward = np.zeros((self.max_size, 1))
		self.not_done = np.zeros((self.max_size, 1))

		self.prioritized = prioritized

		if self.prioritized:
			self.tree = SumTree(self.max_size)
			self.max_priority = 1.0
			self.beta = 0.4


	def append(self, state, action, next_state, reward, done):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done

		if self.prioritized:
			self.tree.set(self.ptr, self.max_priority)
		
		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)


	def sample(self):
		ind = self.tree.sample(self.batch_size) if self.prioritized \
			else np.random.randint(0, self.size, size=self.batch_size)

		batch = (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.LongTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.not_done[ind]).to(self.device)
		)

		if self.prioritized:
			weights = np.array(self.tree.nodes[-1][ind]) ** -self.beta
			weights /= weights.max()
			self.beta = min(self.beta + 2e-7, 1) # Hardcoded: 0.4 + 2e-7 * 3e6 = 1.0. Only used by PER.
			batch += (ind, torch.FloatTensor(weights).to(self.device).reshape(-1, 1))
    
		return batch + (None, None)


	def update_priority(self, ind, priority):
		self.max_priority = max(priority.max(), self.max_priority)
		self.tree.batch_set(ind, priority)

class SumTree(object):
	def __init__(self, max_size):
		self.nodes = []
		# Tree construction
		# Double the number of nodes at each level
		level_size = 1
		for _ in range(int(np.ceil(np.log2(max_size))) + 1):
			nodes = np.zeros(level_size)
			self.nodes.append(nodes)
			level_size *= 2


	# Batch binary search through sum tree
	# Sample a priority between 0 and the max priority
	# and then search the tree for the corresponding index
	def sample(self, batch_size):
		query_value = np.random.uniform(0, self.nodes[0][0], size=batch_size)
		node_index = np.zeros(batch_size, dtype=int)
		
		for nodes in self.nodes[1:]:
			node_index *= 2
			left_sum = nodes[node_index]
			
			is_greater = np.greater(query_value, left_sum)
			# If query_value > left_sum -> go right (+1), else go left (+0)
			node_index += is_greater
			# If we go right, we only need to consider the values in the right tree
			# so we subtract the sum of values in the left tree
			query_value -= left_sum * is_greater
		
		return node_index


	def set(self, node_index, new_priority):
		priority_diff = new_priority - self.nodes[-1][node_index]

		for nodes in self.nodes[::-1]:
			np.add.at(nodes, node_index, priority_diff)
			node_index //= 2


	def batch_set(self, node_index, new_priority):
		# Confirm we don't increment a node twice
		node_index, unique_index = np.unique(node_index, return_index=True)
		priority_diff = new_priority[unique_index] - self.nodes[-1][node_index]
		
		for nodes in self.nodes[::-1]:
			np.add.at(nodes, node_index, priority_diff)
			node_index //= 2
    
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
        self.prioritized = config['prioritized'] if 'prioritized' in config.keys() else True
        self.buffer = PrioritizedStandardBuffer(6, self.batch_size, buffer_size, device, self.prioritized)
        self.epsilon_max = config['epsilon_max'] if 'epsilon_max' in config.keys() else 1.
        self.epsilon_min = config['epsilon_min'] if 'epsilon_min' in config.keys() else 0.01
        self.epsilon_stop = config['epsilon_decay_period'] if 'epsilon_decay_period' in config.keys() else 1000
        self.epsilon_delay = config['epsilon_delay_decay'] if 'epsilon_delay_decay' in config.keys() else 20
        self.epsilon_step = (self.epsilon_max-self.epsilon_min)/self.epsilon_stop
        self.n_step_return = config['n_step_return'] if 'n_step_return' in config.keys() else 1
        self.model = model 
        self.target_model = deepcopy(self.model).to(device)
        self.criterion = config['criterion'] if 'criterion' in config.keys() else torch.nn.MSELoss()
        lr = config['learning_rate'] if 'learning_rate' in config.keys() else 0.001
        self.optimizer = config['optimizer'] if 'optimizer' in config.keys() else torch.optim.Adam(self.model.parameters(), lr=lr)
        self.nb_gradient_steps = config['gradient_steps'] if 'gradient_steps' in config.keys() else 1
        self.update_target_strategy = config['update_target_strategy'] if 'update_target_strategy' in config.keys() else 'replace'
        self.update_target_freq = config['update_target_freq'] if 'update_target_freq' in config.keys() else 20
        self.update_target_tau = config['update_target_tau'] if 'update_target_tau' in config.keys() else 0.005
        

    def gradient_step(self):
        if self.buffer.size > self.batch_size:
            state, action, next_state, reward, done, ind, weights = self.buffer.sample()
            with torch.no_grad():
                next_action = self.model(next_state).argmax(1, keepdim=True)
                update = (
                    reward + done * self.gamma *
                    self.target_model(next_state).gather(1, next_action).reshape(-1, 1)
                )
                
            current_Q = self.model(state).gather(1, action)

            # Compute Q loss
            if self.buffer.prioritized:
                Q_loss = (weights * self.criterion(current_Q, update)).mean()
            else:
                Q_loss = (self.criterion(current_Q, update)).mean()
            # Optimize the Q network
            self.optimizer.zero_grad()
            Q_loss.backward()
            self.optimizer.step()
            if self.buffer.prioritized:
                priority = ((current_Q - update).abs() + 1e-10).pow(0.6).cpu().data.numpy().flatten()
                self.buffer.update_priority(ind, priority)
        
    
    def train(self, env, max_episode):
        episode_return = []
        episode = 0
        episode_cum_reward = 0
        state, _ = env.reset()
        epsilon = self.epsilon_max
        step = 0
        best_model = 0
        memory = deque()
        
        while episode < max_episode:
            # update epsilon
            if step > self.epsilon_delay:
                epsilon = max(self.epsilon_min, epsilon-self.epsilon_step)
            # select epsilon-greedy action
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = greedy_action(self.model, state)
            # step
            next_state, reward, done, trunc, _ = env.step(action)
            # self.buffer.append(state, action, next_state, reward, done)
            memory.append((state, action, next_state, reward, done))
            
            episode_cum_reward += reward
            
            # while len(memory) >= self.n_step_return or (memory and memory[-1][4]):
            while len(memory) >= self.n_step_return : # Not the last one because the end is truncation ? 
                s_mem, a_mem, si_, discount_R, done_ = memory.popleft()
                if not done_ and memory:
                    for i in range(self.n_step_return-1):
                        si, ai, si_, ri, done_ = memory[i]
                        discount_R += ri * self.gamma ** (i + 1)
                        if done_:
                            break
                # self.buffer.append(state, action, next_state, reward, done)
                self.buffer.append(s_mem, a_mem, si_, discount_R, 1 if not done_ else 0)

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
                        ", batch size ", '{:4d}'.format(self.buffer.size), 
                        ", ep return ", locale.format_string('%d', int(episode_cum_reward), grouping=True),
                        sep='')
                    
                score_agent: float = evaluate_HIV(agent=self, nb_episode=1)
                print(locale.format_string('%d', int(score_agent), grouping=True))
                if score_agent > best_model:
                    agent.save()
                    best_model = score_agent
                
                state, _ = env.reset()
                episode_cum_reward = 0
            else:
                state = next_state
            
        return episode_return
    
    def act(self, observation, use_random=False):
        if use_random:
            return np.random.choice(self.env.action_space.n)
        return greedy_action(self.model, observation)
    
    def save(self, path = "prioritez_replay.pt"):
        torch.save({
                    'model_state_dict': self.model.state_dict()
                    }, path)

    def load(self):
        path = "prioritez_replay.pt"
        self.model = DQM_model(6*config['n_state_to_agg'], 256, 4, 6).to(device)
        self.model.load_state_dict(torch.load(path)['model_state_dict'])
    
# Declare network
state_dim = env.observation_space.shape[0]
n_action = env.action_space.n 
nb_neurons= 24


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

model = DQM_model(6, 256, 4, 6).to(device)
# DQN config
config = {'nb_actions': env.action_space.n,
          'learning_rate': 0.001,
          'gamma': 0.99,
          'buffer_size': 100_000,
          'epsilon_min': 0.01,
          'epsilon_max': 1.,
          'epsilon_decay_period': 30_000,
          'epsilon_delay_decay': 5_000,
          'batch_size': 500,
          'gradient_steps': 3,
          'update_target_strategy': 'replace', # or 'ema'
          'update_target_freq': 400,
          'update_target_tau': 0.005,
          'prioritized': False,
          'n_step_return': 5,
          'criterion': torch.nn.SmoothL1Loss()
          }

# Train agent
agent = dqn_agent(config, model)
ep_length = agent.train(env, 2000)
