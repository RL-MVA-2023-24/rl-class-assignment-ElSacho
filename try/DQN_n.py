import random
import gym
import numpy as np
# import cv2
from collections import deque
from tqdm import tqdm
from agent import Agent
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
) 

state_size = 6
action_size = 4
agent = Agent(state_size, action_size, 5) # 5-step return

episodes = 20000
batch_size = 32
total_time = 0 
all_rewards = 0
done = False

# Initializing Buffer
while len(agent.buffer) < 10000:
  state , _ = env.reset()


  for time in range(10000):
      action = agent.act(state)
      next_state, reward, done, trunc, _ = env.step(action)
      
      td_error = agent.calculate_td_error(state, action, reward, next_state, done)
      
      agent.store(state, action, reward, next_state, done, td_error)

      state = next_state

      if done:
          break
  
print("buffer initialized")

for e in tqdm(range(0, episodes)):
    total_reward = 0
    game_score = 0
    state , _ = env.reset()
    frame_stack = deque(maxlen=4)
    frame_stack.append(state)
    
    for skip in range(90):
        env.step(0)
    
    for time in range(20000):
        total_time += 1
        
        if total_time % agent.update_rate == 0:
            agent.update_target_model()
        
        state = sum(frame_stack)/len(frame_stack)
        
        action = agent.act(np.expand_dims(state.reshape(88, 80, 1), axis=0))
        next_state, reward, done, trunc, _ = env.step(action)
        
        frame_stack.append(next_state)
        next_state = sum(frame_stack)/len(frame_stack)
        
        td_error = agent.calculate_td_error(state, action, reward, next_state, done)

        agent.store(state, action, reward, next_state, done, td_error)
        
        state = next_state
        total_reward += reward
        
        if done or trunc:
            all_rewards += total_reward
            print("episode: {}/{}, reward: {}, avg reward: {}"
                  .format(e+1, episodes, total_reward, all_rewards/(e+1)))
            break
            
        agent.replay(batch_size)

    if (e+1) % 500 == 0:
      print("model saved on epoch", e)
    #   agent.save("")