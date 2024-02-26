import numpy as np


class ReplayBuffer(object):
    def __init__(self, max_size=1e6):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def _add(self, transition):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = transition
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(transition)

    def add(self, transitions):
        # transform dict of lists into lists of dicts
        num_transitions = len(transitions['observations'])
        transitions_list = [{key: value[i] for key, value in transitions.items()} for i in range(num_transitions)]
        for transition in transitions_list:
            self._add(transition)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)

        batch = {}
        for key in self.storage[0].keys():
            batch[key] = []

        for i in ind:
            for key, value in self.storage[i].items():
                batch[key].append(value)

        for key, value in batch.items():
            batch[key] = np.stack(value)

        return batch
    
    
import collections
from collections import OrderedDict
import math
import numpy as np


class Node:
    def __init__(self, action, reward, obs, state, mcts, depth, done, parent=None):

        self.env = parent.env
        self.action = action  # Action used to go to this state
        self.done = done

        self.is_expanded = False
        self.parent = parent
        self.children = {}
        self.depth = depth

        self.action_space_size = self.env.action_space.n
        self.child_q_value = np.zeros([self.action_space_size], dtype=np.float32)  # Q
        self.child_priors = np.zeros([self.action_space_size], dtype=np.float32)  # P
        self.child_number_visits = np.zeros([self.action_space_size], dtype=np.float32)  # N

        self.reward = reward
        self.obs = obs
        self.state = state

        self.mcts = mcts

    @property
    def number_visits(self):
        return self.parent.child_number_visits[self.action]

    @property
    def v_value(self):
        # get q_values only for actions that have been tried
        q_values = np.extract(self.child_number_visits > 0, self.child_q_value)
        if len(q_values) > 0:
            return np.mean(q_values)
        return 0

    @number_visits.setter
    def number_visits(self, value):
        self.parent.child_number_visits[self.action] = value

    @property
    def q_value(self):
        return self.parent.child_q_value[self.action]

    @q_value.setter
    def q_value(self, value):
        self.parent.child_q_value[self.action] = value
        self.mcts.update_q_value_stats(value)

    def child_Q(self):
        return self.mcts.normalize_q_value(self.child_q_value)

    def child_U(self):
        c1, c2 = self.mcts.params['c1_coefficient'], self.mcts.params['c2_coefficient']
        utility = math.sqrt(self.number_visits) * self.child_priors / (1 + self.child_number_visits)
        utility *= (c1 + math.log((self.number_visits + c2 + 1) / c2))
        return utility

    def best_action(self):
        """
        :return: action
        """
        child_score = self.child_Q() + self.child_U()
        return np.argmax(child_score)

    def select(self):
        current_node = self
        while current_node.is_expanded:
            best_action = current_node.best_action()
            current_node = current_node.get_child(best_action)
        return current_node

    def expand(self, child_priors):
        self.is_expanded = True
        self.child_priors = child_priors

    def get_child(self, action):
        if action not in self.children:
            # call dynamics function g to obtain new state and reward
            self.env.set_state(self.state)
            obs, reward, done, trunc, _ = self.env.step(action)
            next_state = self.env.get_state()

            self.children[action] = Node(
                obs=obs,
                done=done,
                state=next_state,
                action=action,
                depth=self.depth+1,
                parent=self,
                reward=reward,
                mcts=self.mcts,
            )
        return self.children[action]

    def backup(self, value):
        # update leaf node
        discount_return = value
        self.q_value = (self.number_visits * self.q_value + discount_return) / (1 + self.number_visits)
        self.number_visits += 1
        # update all other nodes up to root node
        current = self.parent
        while current.parent is not None:
            discount_return = current.reward + self.mcts.params['gamma'] * discount_return
            current.q_value = (current.number_visits * current.q_value + discount_return) / (1 + current.number_visits)
            current.number_visits += 1
            current = current.parent


class RootParentNode(object):
    def __init__(self, env):
        self.parent = None
        self.child_q_value = collections.defaultdict(float)
        self.child_number_visits = collections.defaultdict(float)
        self.depth = 0
        self.env = env


class MCTS:
    def __init__(self, mcts_param):
        self.params = mcts_param
        self.max_q_value = -math.inf
        self.min_q_value = math.inf

    def update_q_value_stats(self, q_value):
        self.max_q_value = max(self.max_q_value, q_value)
        self.min_q_value = min(self.min_q_value, q_value)

    def normalize_q_value(self, q_value):
        if self.max_q_value > self.min_q_value:
            return (q_value - self.min_q_value) / (self.max_q_value - self.min_q_value)
        else:
            return q_value

    def compute_priors_and_value(self, node):
        env = node.env
        env.set_state(node.state)

        value = 0.0
        done = False
        t = 0
        while not done:
            _, reward, done, _ = env.step(env.action_space.sample())
            value += reward * (self.params['gamma']**t)
            t += 1

        priors = np.ones((1, node.action_space_size)) / node.action_space_size
        return priors, value

    def compute_action(self, node):
        for _ in range(self.params['num_simulations']):
            leaf = node.select()
            if leaf.done:
                value = leaf.reward
            else:
                child_priors, value = self.compute_priors_and_value(leaf)
                leaf.expand(child_priors)
            leaf.backup(value)

        # Tree policy target (TPT)
        tree_policy = node.child_number_visits / node.number_visits
        tree_policy = tree_policy / np.max(tree_policy)  # to avoid overflows when computing softmax
        tree_policy = np.power(tree_policy, 1/self.params['temperature'])
        tree_policy = tree_policy / np.sum(tree_policy)

        # Choose action according to tree policy
        action = np.random.choice(np.arange(node.action_space_size), p=tree_policy)
        return tree_policy, action, node.v_value, node.children[action]
    
from numpy.polynomial.polynomial import polyval
from scipy.linalg import hankel
import numpy as np
import torch


def compute_support_torch(value, min_val, max_val, mesh=1.0):
    # rescale according to mesh
    value = value / mesh
    min_val = int(min_val / mesh)
    max_val = int(max_val / mesh)

    # value is assumed to be a batch of values of shape (bs,)
    minv = min_val * torch.ones_like(value)
    value = torch.clamp(value, min_val, max_val)

    support_size = max_val - min_val + 1
    support = torch.zeros((value.size()[0], support_size))

    ind = torch.unsqueeze(torch.floor(value) - minv, dim=1)
    src = torch.unsqueeze(torch.ceil(value) - value, dim=1)
    support.scatter_(dim=1, index=ind.long(), src=src)

    ind = torch.unsqueeze(torch.ceil(value) - minv, dim=1)
    src = torch.unsqueeze(value - torch.floor(value), dim=1)
    src.apply_(lambda x: 1 if x == 0 else x)
    support.scatter_(dim=1, index=ind.long(), src=src)

    return support


def compute_cross_entropy(labels, predictions):
    # assume both of shape (bs, num_probs) and to be distributions along last axis
    # assume predictions are raw logits
    predictions = torch.nn.LogSoftmax(dim=-1)(predictions)
    return torch.mean(-torch.sum(labels * predictions, dim=-1))


def compute_value_from_support_torch(support, min_val, max_val, mesh=1.0):
    min_val = int(min_val / mesh)
    max_val = int(max_val / mesh)
    return torch.sum(support * torch.arange(min_val, max_val + 1), dim=1) * mesh


def scaling_func(x, mode='numpy', epsilon=0.001):
    assert mode in ['numpy', 'torch'], 'mode {} not implemented'.format(mode)
    if mode == 'numpy':
        return np.sign(x) * (np.sqrt(np.abs(x) + 1) - 1 + epsilon * x)
    else:
        return torch.sign(x) * (torch.sqrt(torch.abs(x) + 1) - 1 + epsilon * x)


def scaling_func_inv(x, mode='numpy', epsilon=0.001):
    assert mode in ['numpy', 'torch'], 'mode {} not implemented'.format(mode)
    if mode == 'numpy':
        f_ = (np.power((np.sqrt(1 + 4 * epsilon * (np.abs(x) + 1 + epsilon)) - 1) / (2 * epsilon), 2) - 1)
        return np.sign(x) * f_
    else:
        f_ = (torch.pow((torch.sqrt(1 + 4 * epsilon * (torch.abs(x) + 1 + epsilon)) - 1) / (2 * epsilon), 2) - 1)
        return torch.sign(x) * f_


def compute_discounted_n_steps(gamma, rewards, n_steps):
    return polyval(gamma, hankel(rewards)[:n_steps, :])


def compute_td_target(gamma, rewards, values, n_steps):
    assert rewards.shape == values.shape, 'rewards and values of different shape'
    if n_steps < values.shape[0]:
        values = np.concatenate([values[n_steps:], np.zeros((n_steps,))])
        values *= gamma**n_steps
        return compute_discounted_n_steps(gamma, rewards, n_steps) + values
    else:
        return compute_discounted_n_steps(gamma, rewards, n_steps)


def get_temperature_from_schedule(schedule, timesteps):
    t_limits, temps = zip(*schedule)
    t_limits = np.array(t_limits)
    # assume schedule sorted by increasing timesteps
    largers = np.extract(t_limits > timesteps, t_limits)
    if len(largers) > 0:
        idx = np.argmin(np.abs(timesteps - largers)) + t_limits.shape[0] - largers.shape[0]
    else:
        idx = len(t_limits) - 1
    return temps[idx]