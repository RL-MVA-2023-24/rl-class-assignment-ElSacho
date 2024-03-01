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
import time
import torch.nn as nn
from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter
import locale
from collections import deque
locale.setlocale(locale.LC_ALL, 'fr_FR')  # Définir la locale en français
from models import DQM_model
from models import DQM_model, Network, DuelingModel, DistribNet, RainbowNet
import torch.nn.functional as F
import sys
import json
import gymnasium as gym

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PrioritizedStandardBuffer():
	def __init__(self, state_dim, batch_size, buffer_size, device, prioritized, max_priority = 1.0, beta = 0.4):
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
			self.max_priority = max_priority
			self.beta = beta


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
    
		return batch if self.prioritized else batch + (None, None) 


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
        self.device = device
        self.config = config
        self.nb_actions = config['nb_actions']
        self.obs_space = config['obs_space']
        self.gamma = config['gamma'] if 'gamma' in config.keys() else 0.99
        self.batch_size = config['batch_size'] if 'batch_size' in config.keys() else 100
        buffer_size = config['buffer_size'] if 'buffer_size' in config.keys() else int(1e5)
        self.prioritized = config['prioritized'] if 'prioritized' in config.keys() else True
        self.buffer = PrioritizedStandardBuffer(self.obs_space, self.batch_size, buffer_size, device, self.prioritized, config.get('priority', None), config.get('beta', None) )      
        self.epsilon_max = config['epsilon_max'] if 'epsilon_max' in config.keys() else 1.
        self.epsilon_min = config['epsilon_min'] if 'epsilon_min' in config.keys() else 0.01
        self.epsilon_stop = config['epsilon_decay_period'] if 'epsilon_decay_period' in config.keys() else 1000
        self.epsilon_delay = config['epsilon_delay_decay'] if 'epsilon_delay_decay' in config.keys() else 20
        self.epsilon_step = (self.epsilon_max-self.epsilon_min)/self.epsilon_stop
        self.n_step_return = config['n_step_return'] if 'n_step_return' in config.keys() else 1
        self.model = model 
        self.double = config['double'] if 'double' in config.keys() else False
        self.distributional = config['distributional'] if 'distributional' in config.keys() else False
        self.n_atoms = config['n_atoms'] if 'n_atoms' in config.keys() else 50
        self.v_min = config['v_min'] if 'v_min' in config.keys() else 0
        self.v_max = config['v_max'] if 'v_max' in config.keys() else 200
        self.noisy = config['noisy'] if 'noisy' in config.keys() else False
        self.normalize = config['normalize'] if 'normalize' in config.keys() else False
        self.support = torch.linspace(v_min, v_max, n_atoms).to(device) if self.distributional else None
        self.target_model = deepcopy(self.model).to(device)
        self.criterion = config['criterion'] if 'criterion' in config.keys() else torch.nn.MSELoss()
        lr = config['learning_rate'] if 'learning_rate' in config.keys() else 0.001
        self.optimizer = config['optimizer'] if 'optimizer' in config.keys() else torch.optim.Adam(self.model.parameters(), lr=lr)
        if self.double:
            self.optimizer_target = config['optimizer'] if 'optimizer' in config.keys() else torch.optim.Adam(self.model.parameters(), lr=lr)
        self.nb_gradient_steps = config['gradient_steps'] if 'gradient_steps' in config.keys() else 1
        self.update_target_strategy = config['update_target_strategy'] if 'update_target_strategy' in config.keys() else 'replace'
        self.update_target_freq = config['update_target_freq'] if 'update_target_freq' in config.keys() else 20
        self.update_target_tau = config['update_target_tau'] if 'update_target_tau' in config.keys() else 0.005
        self.state_mean = np.array([350_000, 7_750, 300, 30, 37_000, 50], dtype=np.float32)
        self.state_std = np.array([125_000, 15_000, 350, 25, 70_000, 30], dtype=np.float32)
        self.reward_mean = 100_000
        self.reward_std = 300_000
        
    def unormalize_reward(self, reward):
        if self.normalize:
            return reward * self.reward_std + self.reward_mean
        return reward
    
    def normalize_state(self, state):
        return (state - self.state_mean ) / self.state_std
    
    def normalize_reward(self, reward):
        return (reward - self.reward_mean) / self.reward_std
    
    def gradient_step(self):
        if self.buffer.size > self.batch_size:
            state, action, next_state, reward, done, ind, weights = self.buffer.sample()
            # TODO : Tout mettre dans la même brannche quand tout sera re implmenté
            if self.distributional:
                self.loss_distributional(state, action, next_state, reward, done, ind, weights) 
                if self.noisy:
                    self.model.reset_noise()
                    self.target_model.reset_noise()
            elif self.double:
                self.compute_loss_double(state, action, next_state, reward, done, ind, weights) 
                if self.noisy:
                    self.model.reset_noise()
                    self.target_model.reset_noise()
            else :
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
                
                if self.noisy:
                    self.model.reset_noise()
                    self.target_model.reset_noise()

    def compute_loss_double(self, states, actions, next_states, rewards, dones, ind, weights):     
        # compute loss
        curr_Q1 = self.model.forward(states).gather(1, actions)
        curr_Q2 = self.target_model.forward(states).gather(1, actions)

        # next_Q1 = self.model.forward(next_states)
        # next_Q2 = self.target_model.forward(next_states)
        next_Q = torch.min(
            torch.max(self.model.forward(next_states), 1)[0],
            torch.max(self.target_model.forward(next_states), 1)[0]
        )
        next_Q = next_Q.view(next_Q.size(0), 1)
        expected_Q = rewards + dones * self.gamma * next_Q

        if self.prioritized:
            loss1 = (weights * self.criterion(curr_Q1, expected_Q.detach())).mean()
            loss2 = (weights * self.criterion(curr_Q2, expected_Q.detach())).mean()
        else:
            loss1 = self.criterion(curr_Q1, expected_Q.detach()).mean()
            loss2 = self.criterion(curr_Q2, expected_Q.detach()).mean()
        self.optimizer.zero_grad()
        loss1.backward()
        self.optimizer.step()

        self.optimizer_target.zero_grad()
        loss2.backward()
        self.optimizer_target.step()
        
        if self.buffer.prioritized:
            priority = ((curr_Q1 - next_Q).abs() + 1e-10).pow(0.6).cpu().data.numpy().flatten() + ((curr_Q2 - next_Q).abs() + 1e-10).pow(0.6).cpu().data.numpy().flatten()
            self.buffer.update_priority(ind, priority)
            
    def double_distributional(self, states, actions, next_states, rewards, dones, ind, weights):
        # Categorical DQN algorithm
        delta_z = float(self.v_max - self.v_min) / (self.atom_size - 1)
        return
        with torch.no_grad():
            # Double DQN
            next_action = self.dqn(next_state).argmax(1)
            next_dist = self.dqn_target.dist(next_state)
            next_dist = next_dist[range(self.batch_size), next_action]

            t_z = reward +  dones * self.gamma * self.support
            t_z = t_z.clamp(min=self.v_min, max=self.v_max)
            b = (t_z - self.v_min) / delta_z
            l = b.floor().long()
            u = b.ceil().long()

            offset = (
                torch.linspace(
                    0, (self.batch_size - 1) * self.atom_size, self.batch_size
                ).long()
                .unsqueeze(1)
                .expand(self.batch_size, self.atom_size)
                .to(self.device)
            )

            proj_dist = torch.zeros(next_dist.size(), device=self.device)
            proj_dist.view(-1).index_add_(
                0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1)
            )
            proj_dist.view(-1).index_add_(
                0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1)
            )

        dist = self.dqn.dist(state)
        log_p = torch.log(dist[range(self.batch_size), action])
        elementwise_loss = -(proj_dist * log_p).sum(1)

        return elementwise_loss
        
        
        delta_z = float(self.v_max - self.v_min) / (self.n_atoms - 1)

        with torch.no_grad():
            next_action = self.model(next_states).argmax(1)
            next_dist = self.target_model.dist(next_states)
            next_dist = next_dist[range(self.batch_size), next_action]

            t_z = rewards + dones * self.gamma * self.support
            t_z = t_z.clamp(min=self.v_min, max=self.v_max)
            b = (t_z - self.v_min) / delta_z
            l = b.floor().long()
            u = b.ceil().long()

            offset = (
                torch.linspace(
                    0, (self.batch_size - 1) * self.n_atoms, self.batch_size
                ).long()
                .unsqueeze(1)
                .expand(self.batch_size, self.n_atoms)
                .to(self.device)
            )

            proj_dist = torch.zeros(next_dist.size(), device=self.device)
            proj_dist.view(-1).index_add_(
                0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1)
            )
            proj_dist.view(-1).index_add_(
                0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1)
            )

        dist = self.model.dist(states)
        log_p = torch.log(dist[range(self.batch_size), actions])
        elementwise_loss = -(proj_dist * log_p).sum(1)
        loss = elementwise_loss.mean()
        
        if self.buffer.prioritized:
            priority = ((elementwise_loss).abs() + 1e-10).pow(0.6).cpu().data.numpy().flatten()
            self.buffer.update_priority(ind, priority)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def select_action(self, state, step, eval=False):
        if not self.noisy:
            if step > self.epsilon_delay:
                self.epsilon = max(self.epsilon_min, self.epsilon-self.epsilon_step)
            # select epsilon-greedy action
            if np.random.rand() < self.epsilon and not eval:
                action = env.action_space.sample()
            else:
                action = greedy_action(self.model, state)
        else:
            # action = self.model(state).argmax()
            action = self.model(torch.FloatTensor(state).to(self.device)).argmax()
            action = action.detach().cpu().numpy()
        return action
    
    def loss_distributional(self, states, actions, next_states, rewards, dones, ind = None, weights = None):
        """Return categorical dqn loss."""    
        # Categorical DQN algorithm
        delta_z = float(self.v_max - self.v_min) / (self.n_atoms - 1)

        with torch.no_grad():
            next_action = self.model(next_states).argmax(1)
            next_dist = self.target_model.dist(next_states)
            next_dist = next_dist[range(self.batch_size), next_action]

            t_z = rewards + dones * self.gamma * self.support
            t_z = t_z.clamp(min=self.v_min, max=self.v_max)
            b = (t_z - self.v_min) / delta_z
            l = b.floor().long()
            u = b.ceil().long()

            offset = (
                torch.linspace(
                    0, (self.batch_size - 1) * self.n_atoms, self.batch_size
                ).long()
                .unsqueeze(1)
                .expand(self.batch_size, self.n_atoms)
                .to(self.device)
            )

            proj_dist = torch.zeros(next_dist.size(), device=self.device)
            proj_dist.view(-1).index_add_(
                0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1)
            )
            proj_dist.view(-1).index_add_(
                0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1)
            )

        dist = self.model.dist(states)
        log_p = torch.log(dist[range(self.batch_size), actions])

        elementwise_loss = -(proj_dist * log_p).sum(1)
        if self.prioritized:
            loss = (weights * elementwise_loss).mean()
        else:
            loss = elementwise_loss.mean()
            
        if self.buffer.prioritized:
            priority = ((elementwise_loss).abs() + 1e-10).pow(0.6).cpu().data.numpy().flatten()
            self.buffer.update_priority(ind, priority)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def train(self, env, max_episode):
        timestamp = time.strftime("%Y-%m-%d--%H%M")
        expt_name = self.config["save_path"][:-3]
        print("Saving with the name ", expt_name)
        writer = SummaryWriter(f'logs/{expt_name}-{timestamp}')
        episode_return = []
        episode = 0
        episode_cum_reward = 0
        state, _ = env.reset()
        if self.normalize:
            state = self.normalize_state(state)
        self.epsilon = self.epsilon_max
        step = 0
        best_model = 0
        memory = deque()
        
        while episode < max_episode:
            # update epsilon
            action = self.select_action(state, step)
            # step
            next_state, reward, done, trunc, _ = env.step(action)
            episode_cum_reward += reward
            if self.normalize:
                next_state = self.normalize_state(next_state)
                reward = self.normalize_reward(reward)
            if self.n_step_return == 1:
                self.buffer.append(state, action, next_state, reward, done)
            else:
                memory.append((state, action, next_state, reward, done))
                
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
                writer.add_scalar('ep_rew_actual', episode_cum_reward, episode)
                print("Episode ", '{:2d}'.format(episode), 
                        ", epsilon ", '{:6.2f}'.format(self.epsilon), 
                        ", batch size ", '{:4d}'.format(self.buffer.size), 
                        ", ep return ", locale.format_string('%d', int(episode_cum_reward), grouping=True),
                        sep='')
                try:
                    score_agent: float = evaluate_HIV(agent=self, nb_episode=1)
                    print("Result : ", locale.format_string('%d', int(score_agent), grouping=True), " and best result yet : ", locale.format_string('%d', int(best_model), grouping=True))
                    writer.add_scalar('ep_rew_default', score_agent, episode)
                    if score_agent > best_model:
                        agent.save()
                        best_model = score_agent
                except:
                    pass
                    # print("Could not test in the HIV; maybe it is Cartpole")
                state, _ = env.reset()
                if self.normalize:
                    state = self.normalize_state(state)
                episode_cum_reward = 0
            else:
                state = next_state
            
        return episode_return
    
    def act(self, observation, use_random=False):
        if use_random:
            return np.random.choice(self.env.action_space.n)
        if self.normalize:
            observation = self.normalize_state(observation)
        return self.select_action(observation, 1, True)
    
    def save(self, path = "distributional.pt"):
        torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'conf' : self.config
                    }, self.config['save_path'])

    def load(self):
        path = "prioritez_replay.pt"
        self.model = DQM_model(6*config['n_state_to_agg'], 256, 4, 6).to(device)
        self.model.load_state_dict(torch.load(path)['model_state_dict'])
    
if __name__ == "__main__":
    
    env = TimeLimit(
        env=HIVPatient(domain_randomization=True), max_episode_steps=200
    )  

    state_dim = env.observation_space.shape[0]
    n_action = env.action_space.n 

    try : 
        config_path = sys.argv[1]
        with open(config_path, 'r') as file:
            config = json.load(file)
        if config['criterion'] == "SmoothL1Loss":
            config['criterion'] = torch.nn.SmoothL1Loss()
        else:
            print("Add the criterion")
    except:
        config = {'nb_actions': env.action_space.n,
                'obs_space': env.observation_space.shape[0],
                'learning_rate': 0.0005,
                'gamma': 0.999,
                'buffer_size': 200_000,
                'epsilon_min': 0.01,
                'epsilon_max': 1.,
                'epsilon_decay_period': 50_000,
                'epsilon_delay_decay': 2_000,
                'batch_size': 1000,
                'gradient_steps': 3,
                'update_target_strategy': 'replace', # or 'ema'
                'update_target_freq': 600,
                'update_target_tau': 0.005,
                'n_layers' : 6,
                'prioritized': False,
                'priority' : 1.0,
                'beta' : 0.4,
                'n_step_return': 5,
                'noisy': True,
                'noisy_std_init' : 1.5,
                'double' : True,
                'dueling' : True,
                'distributional' : False,
                'v_min' : 0,
                'v_max' : 3e3,
                'n_atoms' : 75,
                'normalize' : True,
                'criterion': torch.nn.SmoothL1Loss(),
                'save_path' : 'noisy_1_5.pt'
                }

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

    # Train agent
    agent = dqn_agent(config, model)
    ep_return = agent.train(env, 2000000)

    # # # DOUBLE         - > OK but not with Distributional
    # # # N-STEP         -> OK
    # # # DUELING        -> OK
    # # # PER            -> OK 
    # # # NOISY          -> OK
    # # # Distributional -> OK But not find good param
