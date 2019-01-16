
import datetime
from unityagents import UnityEnvironment
import random
import copy
import time
from collections import namedtuple, deque
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from agent import Agent, ReplayBuffer, OUNoise

if platform == "linux":
    env = UnityEnvironment(file_name="app/Tennis_Linux")
else:
    env = UnityEnvironment(file_name="app/Tennis.app")

brain_name = env.brain_names[0]
brain = env.brains[brain_name]
env_info = env.reset(train_mode=True)[brain_name]
num_agents = len(env_info.agents)
states = env_info.vector_observations
state_size = states.shape[1]
action_size = brain.vector_action_space_size

BATCH_SIZE = 128        # minibatch size
BUFFER_SIZE = int(1e6)  # replay buffer size
GAMMA = 0.99            # discount factor
LR_ACTOR = 1e-3         # learning rate of the actor
LR_CRITIC = 1e-3        # learning rate of the critic
SEED = 0
TAU = 6e-2              # for soft update of target parameters
WEIGHT_DECAY = 0        # L2 weight decay
UPDATE_EVERY = 1        # time steps between network updates
#N_UPDATES = 1           # number of times training
ADD_NOISE = True

#eps_start = 6           # Noise level start
#eps_end = 0             # Noise level end
#eps_decay = 250         # Number of episodes to decay over from start to end

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("DEVICE:{}".format(device))

""" Setup two independent agents with shared experience memory """
agent_0 = Agent(state_size, action_size, num_agents=1, seed=SEED)
agent_1 = Agent(state_size, action_size, num_agents=1, seed=SEED)


n_episodes = 1000
scores_window = deque(maxlen=100)
scores_all = []
rolling_average = []
elapsed_time_list = []

list1=[]
list2=[]
list3=[]

for i_episode in range(1, n_episodes+1):
    start_time = time.time()
    env_info = env.reset(train_mode=True)[brain_name]
    states = env_info.vector_observations
    states = np.reshape(states, (1, 48))
    agent_0.reset()
    agent_1.reset()
    scores = np.zeros(num_agents)

    while True:
        # Act
        action_0 = agent_0.act(states, ADD_NOISE)
        action_1 = agent_1.act(states, ADD_NOISE)
        actions = np.concatenate((action_0, action_1), axis=1)
        env_info = env.step(actions)[brain_name]
        next_states = env_info.vector_observations
        next_states = np.reshape(next_states, (1, 48))
        rewards = env_info.rewards
        done = env_info.local_done

        # Learn
        agent_0.step(states, actions, rewards[0], next_states, done, 0)
        agent_1.step(states, actions, rewards[1], next_states, done, 1)

        # Update scores
        scores += rewards

        # Roll over states to next time step
        states = next_states

        # Break if done
        if np.any(done):
            break

    scores_window.append(np.max(scores))
    scores_all.append(np.max(scores))
    rolling_average.append(np.mean(scores_window))
    
    elapsed_time = time.time() - start_time
    elapsed_time_list.append(elapsed_time)

    if i_episode % 10 == 0:
        list1.append(np.max(scores))
        list2.append(np.mean(scores_window))
        list3.append(np.mean(elapsed_time_list))
        print('Episode {}\tMax Reward: {:.3f}\tMean Reward: {:.3f}\tMean Episode Train: {:.2f}s'.format(
            i_episode, np.max(scores), np.mean(scores_window), np.mean(elapsed_time_list)))
        elapsed_time_list = []

    if np.mean(scores_window) >= 0.5:
        print('\nEnvironment Solved!')
        print('Total Episodes: {}\t Average Score: {}'.format(i_episode-100, np.mean(scores_window)))

        df = pd.DataFrame({'Episode' : range(i_episode),
                           'Batch Size' : BATCH_SIZE,
                           'fc1' : 128,
                           'fc2' : 64,
                           'Max Reward' : list1,
                           'Mean Reward' : list2,
                           'Mean Episode Train Time (s)' : list3})
        df.to_csv("Train_file_1.csv")
        #torch.save(agent_0.actor_local.state_dict(), 'checkpoint_actor_0.pth')
        #torch.save(agent_0.critic_local.state_dict(),
        #           'checkpoint_critic_0.pth')
        #torch.save(agent_1.actor_local.state_dict(), 'checkpoint_actor_1.pth')
        #torch.save(agent_1.critic_local.state_dict(),
        #           'checkpoint_critic_1.pth')
        break
    if i_episode >= 600:
        test = list(range(0+10, i_episode, 10))
        df = pd.DataFrame({'Episode': list(range(0+10, i_episode+10, 10)),
                           'Batch Size': BATCH_SIZE,
                           'fc1': 128,
                           'fc2': 64,
                           'Max Reward': list1,
                           'Mean Reward': list2,
                           'Mean Episode Train Time (s)': list3})
        df.to_csv("Train_file_1.csv")
        exit()
