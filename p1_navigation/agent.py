import numpy as np
import random
from collections import namedtuple, deque

from model import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

class DQN_Agent():
    """Uses a classic Deep Q-Network to learn from the environment"""

    def __init__(self, state_size, action_size, seed, device, args):
        super(DQN_Agent, self).__init()

        #initialize network parameters
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.lr = args.lr
        self.update_every = args.update_every
        self.gamma = args.gamma
        self.tau = args.tau
        self.batchsize = args.batchsize
        self.buffersize = args.buffersize


        #Initialize a Q-Network
        self.qnet_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnet_target = QNetwork(state_size, action_size, seed).to(device)

        #set optimizer
        self.optimizer = optim.Adam(self.qnet_local.parameters(), lr=self.lr)

        #initialize REPLAY memory
        self.memory = ReplayBuffer(action_size, self.buffersize, self.batchsize, seed)

        #initialize time step (for use with the updat_every parameter)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        #save the current SARSA status in the replay memory
        self.memory.add(state, action, reward, next_state, done)

        #learn every "update_every" timesteps
        self.t_step = (self.t_step + 1) % self.update_every
