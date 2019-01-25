import numpy as np
import random
from collections import namedtuple, deque

from model import QNetwork
from agent_utils import ReplayBuffer

import torch
import torch.nn.functional as F
import torch.optim as optim

class DQN_Agent():
    """Uses a classic Deep Q-Network to learn from the environment"""

    def __init__(self, nS, nA, device, args, seed):
        #super(DQN_Agent, self).__init()

        #initialize agent parameters
        self.nS = nS
        self.nA = nA
        self.seed = random.seed(seed)
        self.device = device
        #initialize params from the command line args
        self.batchsize = args.batchsize
        self.buffersize = args.buffersize
        self.epsilon = args.epsilon
        self.gamma = args.gamma
        self.lr = args.learn_rate
        self.tau = args.tau
        self.update_every = args.update_every


        #Initialize a Q-Network
        self.qnet_local = QNetwork(nS, nA, seed).to(device)
        self.qnet_target = QNetwork(nS, nA, seed).to(device)

        #set optimizer
        #SET THIS TO CALL "get_optimizer()" FUNCTION THAT STILL NEEDS TO BE
        #WRITTEN IN THE FUTURE, BUT SO THAT DIFFERENCE AGENT TYPES CAN ALL CALL
        #A FUNCTION THAT WILL SET DIFFERENT TYPES OF OPTIMS FROM THE CMD LINE

        self.optimizer = optim.Adam(self.qnet_local.parameters(), lr=self.lr)

        #initialize REPLAY memory
        self.memory = ReplayBuffer(nA, self.buffersize, self.batchsize, seed, device)

        #initialize time step (for use with the update_every parameter)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        #save the current SARSA status in the replay memory
        self.memory.add(state, action, reward, next_state, done)

        #once the replay memory accumulates enough samples, then learn every "update_every" timesteps
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0 and len(self.memory) > self.batchsize:
            batch = self.memory.sample()
            self.learn(batch)

    def act(self, state, epsilon):
        #π
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnet_local.eval()
        with torch.no_grad():
            action_values = self.qnet_local(state).cpu().data.numpy()
        self.qnet_local.train()

        #select an action using epsilon-greedy π
        probs = np.ones(self.nA) * epsilon / self.nA
        probs[np.argmax(action_values)] = 1 - epsilon + (epsilon / self.nA)
        return np.random.choice(np.arange(self.nA), p = probs)


    def learn(self, batch):
        states, actions, rewards, next_states, dones = batch

        #get max predicted Q values for the next states from the target model
        target_qnet_nV = self.qnet_target(next_states).detach().max(1)[0].unsqueeze(1)

        #compute Q targets for the current states
        q_targets = rewards + (self.gamma * target_qnet_nV * (1 - dones))

        #get the expected Q values from the local model
        q_expected = self.qnet_local(states).gather(1, actions)

        #compute the loss values over the network
        loss = F.mse_loss(q_expected, q_targets)

        #minimize the loss (backpropogation)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        #update the target network
        self.qnet_update()

    def qnet_update(self):
        #Copy the params from the target network to the local network at rate TAU
        for target_param, local_param in zip(self.qnet_target.parameters(), self.qnet_local.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1-self.tau)*target_param.data)
