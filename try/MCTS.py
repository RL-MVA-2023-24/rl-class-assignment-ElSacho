from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
import random
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from cartpole import CartPole
import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
import locale
locale.setlocale(locale.LC_ALL, 'fr_FR')  # Définir la locale en français

from mcts_utils import MCTS, Node, RootParentNode
import torch
import torch.nn as nn
from mcts_utils import scaling_func_inv, compute_value_from_support_torch, compute_td_target, compute_support_torch, scaling_func, compute_cross_entropy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.

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

class AlphaZeroModel(nn.Module):
    def __init__(self, obs_space, action_space, model_config):

        self.num_actions = action_space.n
        self.obs_dim = obs_space.shape[0]
        self.value_min_val = model_config['value_support_min_val']
        self.value_max_val = model_config['value_support_max_val']
        self.value_support_size = self.value_max_val - self.value_min_val + 1

        nn.Module.__init__(self)

        # Let's use a simple dense model
        num_hidden = model_config['num_hidden']
        self.decision_function_shared = nn.Sequential(
            nn.Linear(in_features=self.obs_dim, out_features=num_hidden),
            nn.ReLU(),
            nn.Linear(in_features=num_hidden, out_features=num_hidden),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(in_features=num_hidden, out_features=self.num_actions)
        self.value_head = nn.Linear(in_features=num_hidden, out_features=self.value_support_size)

    def forward(self, obs):
        x = self.decision_function_shared(obs)
        logits = self.policy_head(x)
        values_support_logits = self.value_head(x)
        return logits, values_support_logits

    def compute_priors_and_value(self, obs):
        with torch.no_grad():
            obs = torch.from_numpy(obs).float()
            #TODO: complete here
            
            # this function takes a batch of numpy observations,
            # computes a batch of priors (as probabilities, not raw logits)
            # and a batch of values (scalar and unscaled with h^{-1})
            logits, value_support = self.forward(obs)
            prior = nn.Softmax(dim=-1)(logits)
            value_support = nn.Softmax(dim=-1)(value_support)
            value = compute_value_from_support_torch(value_support, self.value_min_val, self.value_max_val)
            value = scaling_func_inv(value, mode='torch')
            
            # both priors and values returned are numpy as well
            return prior.cpu().numpy(), value.cpu().numpy()
        
class AlphaZeroMCTS(MCTS):
    def __init__(self, mcts_param, model):
        MCTS.__init__(self, mcts_param)
        self.model = model
        self.config = mcts_param
        
    def add_noise_to_priors(self, priors):
        noise = np.random.dirichlet([self.config['dir_noise']] * priors.size)
        priors = (1 - self.config['dir_epsilon']) * priors
        priors += self.config['dir_epsilon'] * noise
        return priors
    
    def compute_priors_and_value(self, node):
        obs = np.expand_dims(node.obs, axis=0)  # add batch size of 1
        priors, value = self.model.compute_priors_and_value(obs)
        if self.config['add_dirichlet_noise']:
            priors = self.add_noise_to_priors(priors)
        return priors, value
    
    def compute_action(self, node):
        # Run simulations
        for _ in range(self.config['num_simulations']):
            leaf = node.select()
            if leaf.done:
                value = leaf.reward
            else:
                child_priors, value = self.compute_priors_and_value(leaf)
                leaf.expand(child_priors)
            leaf.backup(value)

        # Compute Tree policy target (TPT):
        tree_policy = node.child_number_visits / node.number_visits
        tree_policy = tree_policy / np.max(tree_policy)  # to avoid overflows when computing softmax
        tree_policy = np.power(tree_policy, 1 / self.config['temperature'])
        tree_policy = tree_policy / np.sum(tree_policy)
        
        
        # Compute Tree value
        tree_value = node.v_value
        
        # Choose action according to tree policy
        action = np.random.choice(np.arange(node.action_space_size), p=tree_policy)
        
        return tree_policy, action, tree_value, node.children[action]
    
    
from mcts_utils import ReplayBuffer

class AlphaZero:
    def __init__(self, env_creator, config):
        self.env_creator = env_creator
        self.env = env_creator()
        self.config = config
        self.mcts_config = config['mcts_config']
        self.mcts_config.update(config)
        self.model = AlphaZeroModel(self.env.observation_space, self.env.action_space, config['model_config'])
        self.replay_buffer = ReplayBuffer(config['buffer_size'])
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['lr'])
        self.total_num_steps = 0

    def play_episode(self):
        transitions = {
            "observations": [],
            "actions": [],
            "rewards": [],
            "tree_policies": [],
            "tree_values": [],
        }
        
        # play one episode with mcts and store the observations, actions, rewards, 
        # tree policies and tree values at each timestep in the dictionnary transitions

        obs, _ = self.env.reset()
        env_state = self.env.get_state()
        done = False

        mcts = AlphaZeroMCTS(self.mcts_config, self.model)

        root_node = Node(
            state=env_state,
            done=False,
            obs=obs,
            reward=0,
            action=None,
            parent=RootParentNode(env=self.env_creator()),
            mcts=mcts,
            depth=0
        )

        while not done or not trunc:
            transitions['observations'].append(obs)

            tree_policy, action, tree_value, root_node = mcts.compute_action(root_node)
            root_node.parent = RootParentNode(env=self.env_creator())
            obs, reward, done, trunc, _ = self.env.step(action)

            transitions['actions'].append(action)
            transitions['tree_policies'].append(tree_policy)
            transitions['tree_values'].append(tree_value)
            transitions['rewards'].append(reward)

        return transitions

    def postprocess_transitions(self, transitions):
        # transitions dict flows directly into this function when an episode has been played
        # compute the value targets from the rewards and tree values
        # the parameter gamma is in self.config['gamma'] and the parameter n is in
        # self.config['n_steps']
        value_targets = compute_td_target(
            self.config['gamma'],
            np.array(transitions['rewards']),
            np.array(transitions['tree_values']),
            self.config['n_steps']
        )
        
        # we scale the value targets using function h
        value_targets = scaling_func(value_targets, mode='numpy')
        
        # we transform the np array into a list of numpy arrays, one per transition
        transitions['value_targets'] = np.split(value_targets, len(value_targets))

        # we dont store useless arrays in the buffer
        del transitions['rewards']
        del transitions['tree_values']

        return transitions

    def compute_loss(self, batch):
        # compute AlphaZero loss in this function
        # batch is a dict of transitions with keys: 'observations', 'tree_policies', 'value_targets'
        # each key is associated to a numpy which first dim equals batch size
        
        # first we get supports parameters
        v_support_minv, v_support_maxv = self.model.value_min_val, self.model.value_max_val
        
        # transform numpy vectors to torch tensors
        observations = torch.from_numpy(batch['observations']).float()
        mcts_policies = torch.from_numpy(batch["tree_policies"]).float()
        value_targets = torch.from_numpy(batch["value_targets"]).float()[:, 0]
        
        # compute losses
        
        # run model on observations
        logits, values_support_logits = self.model(observations)
        
        # project value oto support
        vtargets_supports = compute_support_torch(value_targets, v_support_minv, v_support_maxv)
        
        # compute cross entropy on policy and value
        policy_loss = compute_cross_entropy(labels=mcts_policies, predictions=logits)
        value_loss = compute_cross_entropy(vtargets_supports, values_support_logits)

        # compute total loss
        # we rescale the value loss with a coefficient given as an hyperparameter
        value_loss = self.config['value_loss_coefficient'] * value_loss
        total_loss = policy_loss + value_loss
        return total_loss, policy_loss, value_loss

    def train(self):
        # we train the agent for several epochs. In this notebook we define an epoch as the succession
        # of data generation (we play episodes with the MCTS) and training (we sample 
        # batches of data in the replay buffer and train on them)
        for _ in range(self.config['num_epochs']):
            episode_rewards = []
            num_steps = 0
            for _ in range(self.config['num_episodes_per_epoch']):
                # play an episode
                transitions = self.play_episode()
                episode_rewards.append(np.sum(transitions['rewards']))
                num_steps += len(transitions['rewards'])
                # process the transitions
                transitions = self.postprocess_transitions(transitions)
                # store them in the replay buffer
                self.replay_buffer.add(transitions)

            avg_rewards = np.mean(episode_rewards)
            max_rewards = np.max(episode_rewards)
            min_rewards = np.min(episode_rewards)
            self.total_num_steps += num_steps

            s = 'Num timesteps sampled so far {}'.format(self.total_num_steps)
            s += ', mean accumulated reward: {}'.format(avg_rewards)
            s += ', min accumulated reward: {}'.format(min_rewards)
            s += ', max accumulated reward: {}'.format(max_rewards)
            print(s)

            # we want for the buffer to contain a minimum numer of transitions
            # if enough timesteps collected, then start training
            if self.total_num_steps >= self.config['learning_starts']:
                
                # perform one SGD per transition sampled
                for _ in range(num_steps):
                    # sample transitions in the replay buffer
                    batch = self.replay_buffer.sample(self.config['batch_size'])
                    # compute loss
                    total_loss, policy_loss, value_loss = self.compute_loss(batch)
                    # do backprop
                    self.optimizer.zero_grad()
                    total_loss.backward()
                    self.optimizer.step()
                    
# create an env_creator function
env_creator = lambda: CartPole()

# define the config with the hyper-parameters
config = {
    'buffer_size': 1000,
    'batch_size': 256,
    'lr': 1e-3,

    'gamma': 0.997,
    'n_steps': 10,

    'num_epochs': 100,
    'num_episodes_per_epoch': 5,
    'learning_starts': 500,  # number of timesteps to sample before SGD

    'value_loss_coefficient': 0.2,

    'model_config': {
        'value_support_min_val': 0,
        'value_support_max_val': 30,
        'num_hidden': 32,
    },

    'mcts_config': {
        'num_simulations': 20,
        "temperature": 1.0,
        "c1_coefficient": 1.25,
        "c2_coefficient": 19652,
        'add_dirichlet_noise': True,
        'dir_noise': 0.5,
        'dir_epsilon': 0.2,
    }
}

# instanciate the agent
agent = AlphaZero(env_creator, config)

# train it
agent.train()