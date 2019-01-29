import numpy as np
import random
from collections import namedtuple, deque
# 
# from model import QNetwork
# from agent_utils import ReplayBuffer

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Agent():
    """Uses a classic Deep Q-Network to learn from the environment"""

    def __init__(self, nS, nA, device, args, seed=0):
        #super(DQN_Agent, self).__init()

        #initialize agent parameters
        self.nS = nS
        self.nA = nA
        self.seed = random.seed(seed)
        self.device = device

        #initialize params from the command line args
        self.framework = args.framework
        self.batchsize = args.batchsize
        self.buffersize = args.buffersize
        self.dropout = args.dropout
        self.epsilon = args.epsilon
        self.epsilon_decay = args.epsilon_decay
        self.epsilon_min = args.epsilon_min
        self.gamma = args.gamma
        self.lr = args.learn_rate
        self.tau = args.tau
        self.update_every = args.update_every
        self.momentum = args.momentum

        #initialize REPLAY memory
        self.memory = ReplayBuffer(nA, self.buffersize, self.batchsize, seed, device)

        #Initialize a Q-Network
        self.q = QNetwork(nS, nA, seed, self.dropout).to(device)
        self.qhat = QNetwork(nS, nA, seed, self.dropout).to(device)
        self.qhat.load_state_dict(self.q.state_dict())

        #set optimizer
        if args.optimizer == "RMSprop":
            self.optimizer = optim.RMSprop(self.q.parameters(), lr=self.lr, momentum=self.momentum)
        elif args.optimizer == "Adam":
            self.optimizer = optim.Adam(self.q.parameters(), lr=self.lr)
        else:
            self.optimizer = optim.SGD(self.q.parameters(), lr=self.lr, momentum=self.momentum)



        #initialize time step (for use with the update_every parameter)
        self.t_step = 0

    def update_epsilon(self):
        self.epsilon = max(self.epsilon*self.epsilon_decay, self.epsilon_min)

    def step(self, state, action, reward, next_state, done):
        #save the current SARS′  status in the replay memory
        self.memory.add(state, action, reward, next_state, done)

        #once the replay memory accumulates enough samples, then learn every "update_every" timesteps
        if len(self.memory) > self.batchsize and self.t_step % self.update_every == 0:
            batch = self.memory.sample()
            self.learn(batch)
        self.t_step += 1

    def act(self, state):
        #send the state to a tensor object on the gpu
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.q.eval() #put network into eval mode
        with torch.no_grad():
            action_values = self.q(state).cpu().data.numpy()
        self.q.train() #put network back into training mode

        #select an action using epsilon-greedy π
        probs = np.ones(self.nA) * self.epsilon / self.nA
        probs[np.argmax(action_values)] = 1 - self.epsilon + (self.epsilon / self.nA)
        return np.random.choice(np.arange(self.nA), p = probs)


    def learn(self, batch):
        states, actions, rewards, next_states, dones = batch

        if self.framework == 'DQN':
            #VANILLA DQN: get max predicted Q values for the next states from the target model
            qhat_nextvalues = self.qhat(next_states).detach().max(1)[0].unsqueeze(1)

        if self.framework == 'DDQN':
            #DOUBLE DQN: get q-value for max action in Q, evaluate under qHAT
            q_next_actions = self.q(next_states).detach().argmax(1).unsqueeze(1)
            qhat_nextvalues = self.qhat(next_states).gather(1, q_next_actions)


        #compute Q targets for the current states using current Q and Q^ of S′
        expected_state_action_values = rewards + (self.gamma * qhat_nextvalues * (1 - dones))

        #get the expected values from the current model Q
        state_action_values = self.q(states).gather(1, actions)

        #compute the loss values over the network
        loss = F.mse_loss(state_action_values, expected_state_action_values)

        #minimize the loss (backpropogation)
        self.optimizer.zero_grad()
        loss.backward()
        # for param in self.q.parameters():
        #     param.grad.data.clamp(-1,1)
        self.optimizer.step()
        #update the target network
        if self.t_step % (self.update_every ** 3 ) == 0:
            self.qhat.load_state_dict(self.q.state_dict())
            #self.qnet_update()



    def qnet_update(self):
        #Copy the params from the target network to the local network at rate TAU
        for target_param, local_param in zip(self.qhat.parameters(), self.q.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1-self.tau)*target_param.data)



class ReplayBuffer:
    def __init__(self, nA, buffersize, batchsize, seed, device):
        self.nA = nA
        self.memory = deque(maxlen=buffersize)
        self.batchsize = batchsize
        self.item = namedtuple("Item", field_names=['state','action','reward','next_state','done'])
        self.seed = random.seed(seed)
        self.device = device

    def add(self, state, action, reward, next_state, done):
        t = self.item(state, action, reward, next_state, done)
        self.memory.append(t)

    def sample(self):
        batch = random.sample(self.memory, k=self.batchsize)

        states = torch.from_numpy(np.vstack([item.state for item in batch if item is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([item.action for item in batch if item is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([item.reward for item in batch if item is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([item.next_state for item in batch if item is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([item.done for item in batch if item is not None]).astype(np.uint8)).float().to(self.device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)



class QNetwork(nn.Module):
    """Deep Q-Network Model. Nonlinear estimator for Qπ
    """

    def __init__(self, state_size, action_size, seed, dropout=0.25, layer_sizes=[64, 64]):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()

        # self.seed = torch.manual_seed(seed)
        # self.fc1 = nn.Linear(state_size, fc1_units)
        # self.fc2 = nn.Linear(fc1_units, fc2_units)
        # self.fc3 = nn.Linear(fc2_units, action_size)

        self.seed = torch.manual_seed(seed)
        self.hidden_layers = nn.ModuleList([nn.Linear(state_size, layer_sizes[0])])
        self.output = nn.Linear(layer_sizes[-1], action_size)

        layer_sizes = zip(layer_sizes[:-1], layer_sizes[1:])
        self.hidden_layers.extend([nn.Linear(i, o) for i, o in layer_sizes])
        self.dropout = nn.Dropout(p=dropout)


    def forward(self, state):
        """Build a network that maps state -> action values."""
        # x = F.relu(self.fc1(state))
        # x = F.relu(self.fc2(x))
        # return self.fc3(x)

        x = F.relu(self.hidden_layers[0](state))
        x = self.dropout(x)
        for layer in self.hidden_layers[1:]:
            x = F.relu(layer(x))
            x = self.dropout(x)
        return self.output(x)
