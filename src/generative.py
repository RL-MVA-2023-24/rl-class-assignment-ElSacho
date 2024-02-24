import torch
from torch import nn
import torch.nn.functional as F
import random
import numpy as np
import pygame
import os
from tqdm import tqdm 

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

HIDDEN_SIZE = 50
GENERATION_SIZE = 100
GOAL_SCORE = 10_000_000_000
MUTATIONS_RATE = [0, 0.2, 0.4, 0.9]
MUTATION_DISTRIBUTION = [0, 0.25, 0.50, 0.75]
MAX_GEN_NUMBER = 200
GENETIC_PATH = "nn"

class GenerationManager():
    def __init__(self, game):
        self.generation_number = 0
        self.score_goal = GOAL_SCORE
        self.generation_size = GENERATION_SIZE
        self.agents = [ NeuralAgent(game, HIDDEN_SIZE) for _ in range(self.generation_size)]
            
    def rank(self):
        self.agents = sorted(self.agents, key=lambda agent: agent.score, reverse=True)

    def mutate(self):
        for i in range(len(self.agents)):
            self.agents[i].reset_score()
            for j in range(len(MUTATIONS_RATE) - 1):
                if i < int(self.generation_size * MUTATION_DISTRIBUTION[j + 1]) and i >= int(self.generation_size * MUTATION_DISTRIBUTION[j]):
                    self.agents[i].mutate(MUTATIONS_RATE[j])
                    break
                if j == len(MUTATIONS_RATE) - 2:
                    self.agents[i].mutate(MUTATIONS_RATE[-1])
                    break
        
    def print_results(self):
        print("\nGeneration ", self.generation_number)
        print(f"The best score is {self.agents[0].score}")
        average_score = sum(agent.score for agent in self.agents) / len(self.agents)
        average_score_25_best = sum(self.agents[i].score for i in range(25)) / 25
        print("The average score is ", average_score)
        print("The average score for the top 25 is ", average_score_25_best)
        
    def show_scores(self):
        for i in range(len(self.agents)):
            print(f"The score of the player {i} is {self.agents[i].score}")
            
    def play(self):
        self.generation_number += 1
        for i in tqdm(range(len(self.agents))):
            random.seed(self.generation_number)
            random.seed(1)
            self.agents[i].play_one_game(draw = False)
            
    def show_best_generation_player(self):
        random.seed(self.generation_number)
        print("seed : ",self.generation_number)
        print('Showing the score of : ', self.agents[0].score)
        self.agents[0].play_one_game(draw = True)
                     
    def train(self):
        self.best_score = 0
        while self.generation_number < MAX_GEN_NUMBER and self.best_score < GOAL_SCORE:
            self.play()
            self.rank()
            self.agents[0].save_model("nn/best_agent_path.pt")  
            self.best_score = max(self.best_score, self.agents[0].score)
            # if self.best_score >= GOAL_SCORE:
            #     if not self.test_best_score() : self.best_score -= 1
            # self.save_nn() 
            self.print_results()
            self.mutate()
        self.print_message_end_training()
        
    def test_best_score(self):
        for agent in self.agents:
            if agent.score >= GOAL_SCORE:
                if agent.test_agent():
                    return True
                else:
                    agent.score -= 1
            else:
                return False
        return False
                
    def print_message_end_training(self):
        if self.best_score >= GOAL_SCORE:
            print("\n============= END OF THE TRAINING ===============")
            if self.generation_number > 1:
                print(f"Goal achieved in {self.generation_number} generations !")
            else :
                print(f"Goal achieved in {self.generation_number} generation ! You could consider putting a higher goal.")
        elif self.generation_number >= MAX_GEN_NUMBER:
            print(f"Unfortunatly, you could only reach the score of {self.best_score} in {self.generation_number} generations..")
            
    def save_nn(self):
        # We create the folder if necessarly
        if not os.path.exists(GENETIC_PATH):
            os.makedirs(GENETIC_PATH)
            
        all_nn_path = os.path.join(GENETIC_PATH, 'all_nn')
        best_nn_path = os.path.join(GENETIC_PATH, 'best_nn')

        if not os.path.exists(all_nn_path):
            os.makedirs(all_nn_path)

        if not os.path.exists(best_nn_path):
            os.makedirs(best_nn_path)
            
        # we save all the models in the folder all_nn with their number and their score
        compt = 0 
        for agent in self.agents:
            agent.save_model(all_nn_path, agent_number = compt)
            compt += 1
        
        # we save the best model if it reached the goal
        self.agents[0].save_model(best_nn_path)  

class NeuralAgent(nn.Module):
    def __init__(self, env, hidden_size):
        super(NeuralAgent, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hidden = nn.Linear(6, hidden_size).to(self.device)
        self.hidden2 = nn.Linear(hidden_size, hidden_size).to(self.device)
        self.output = nn.Linear(hidden_size, env.action_space.n).to(self.device)
        
        self.game = env
        self.score = 0

    def forward(self, x):
        x = torch.from_numpy(x).to(self.device)
        x = x.to(next(self.parameters()).dtype)
        x = F.relu(self.hidden(x))
        x = F.relu(self.hidden2(x))
        x = self.output(x)
        x = x.unsqueeze(0)
        return F.softmax(x, dim=1)
    
    def mutate(self, mutation_rate):
        with torch.no_grad():  # Nous ne voulons pas de gradients pendant la mutation
            for param in self.parameters():
                # Crée un masque de la même taille que le paramètre
                # Le masque est vrai avec une probabilité égale au pourcentage
                mask = torch.rand_like(param) < (mutation_rate)

                # Génère de nouveaux poids aléatoires
                new_weights = torch.randn_like(param)

                # Applique la mutation aux poids sélectionnés
                param[mask] = new_weights[mask]
                
    def play_one_game(self, draw = False):
        # We reset the game to have the state
        next_state, _ = self.game.reset()
        # We check if the score is under the max goal we want, but we underpass this score if we are drawing the player
        while self.score < GOAL_SCORE or draw:
            action = self.forward(next_state)
            action_index = torch.argmax(action)
            
            next_state, reward, done, truc, _ = self.game.step(action_index)
            
            self.score += reward

            if done or truc == True:
                break
        return 
    
    def test_agent(self, draw = False):
        print("Testing the agent ...")
        for seed_value in tqdm(range(1, 20)):
            random.seed(seed_value)
            self.score = 0
            if draw and GameParameters.RENDER_GAMEPLAY:
                # Libérez les ressources et fermez les fenêtres
                self.game.start_video(f'OutputTestAgentInWorld{seed_value}')
                self.play_one_game(draw = draw)
                self.game.make_video(f'OutputTestAgentInWorld{seed_value}.mp4')
            else:
                self.play_one_game(draw = draw)
            print(self.score)
            if self.score < gen.GOAL_SCORE:
                print("Testing failed")
                self.score = gen.GOAL_SCORE - 1
                return False
        return True

    def save_model(self, file_path, agent_number = None):
        # if agent_number == None, we save the best model
        # torch.save(self.state_dict(), file_path)
        torch.save({
                    'model_state_dict': self.target_model.state_dict()
                    }, file_path)
        # if agent_number == None:
        #     torch.save(self.state_dict(), file_path)
        # # otherwise, we save all the models
        # else:
        #     formatted_agent_number = "{:02d}".format(agent_number)
        #     score_format = "{:02d}".format(self.score)
        #     file_path = file_path + "/agent_" + formatted_agent_number + "_score_" + score_format + ".pt"
        #     torch.save(self.state_dict(), file_path)
            
    def load_model(self, file_path):
        self.load_state_dict(torch.load(file_path))
        
    def reset_score(self):
        self.score = 0


if __name__ == "__main__":
    
    pygame.init()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = TimeLimit(
        env=HIVPatient(domain_randomization=False), max_episode_steps=200
    )  # The time wrapper limits the number of steps in an episode at 200.
    # Now is the floor is yours to implement the agent and train it.

    
    generation_manager = GenerationManager(env)
    
    generation_manager.train()
    
    # best_agent = NeuralAgent(env, HIDDEN_SIZE)
    # best_model_path = "genetic_nn/best_nn/score_50.pt"
    # best_agent.load_model( best_model_path )
    
    # draw_test = True
    # print(best_agent.test_agent(draw = False))
    # print(best_agent.test_agent(draw = draw_test))


    