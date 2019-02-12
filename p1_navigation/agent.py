import numpy as np
import random
from collections import namedtuple, deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Agent():
    """Uses a classic Deep Q-Network to learn from the environment"""

    def __init__(self, nA, nS, args, seed=0):
        #super(DQN_Agent, self).__init()

        #initialize agent parameters
        self.nS = nS
        self.nA = nA
        self.seed = 0#random.seed(seed)
        self.framestack = args.framestack
        self.device = args.device
        self.t_step = 0
        self.optimizer = args.optimizer
        #initialize params from the command line args
        self.framework = args.framework
        self.batchsize = args.batchsize
        self.buffersize = args.buffersize
        self.C = args.C
        self.dropout = args.dropout
        self.epsilon = args.epsilon
        self.epsilon_decay = args.epsilon_decay
        self.epsilon_min = args.epsilon_min
        self.gamma = args.gamma
        self.lr = args.learn_rate
        self.update_every = args.update_every
        self.momentum = args.momentum
        self.PER = args.prioritized_replay
        self.train = args.train
        #self.pixels = args.pixels

        #initialize REPLAY buffer
        self.buffer = ReplayBuffer(self.buffersize, self.batchsize, self.framestack, self.device)

        #Initialize Q-Network
        self.q = self._make_model(args.pixels)
        self.qhat = self._make_model(args.pixels)
        self.qhat.load_state_dict(self.q.state_dict())
        self.qhat.eval()
        self.optimizer = self._set_optimizer(self.q.parameters())

    def act(self, state):
        """Select an action using epsilon-greedy π.
           Always use greedy if not training.
        """
        if random.random() > self.epsilon or not self.train:
            self.q.eval()
            with torch.no_grad():
                action_values = self.q(state)
            self.q.train()
            return action_values.max(1)[1].view(1,1)
        else:
            return torch.tensor([[random.randrange(self.nA)]], device=self.device, dtype=torch.long)

    def step(self, state, action, reward, next_state, done):
        """Moves the agent to the next timestep.
           Learns each UPDATE_EVERY steps.
        """
        if not self.train:
            return

        #save the current SARS′  status in the replay buffer
        self.buffer.add(state, action, reward, next_state)

        if len(self.buffer) >= self.batchsize and self.t_step % self.update_every == 0:
            self.learn()
        #update the target network every C steps
        if self.t_step % self.C == 0:
            self.qhat.load_state_dict(self.q.state_dict())
        self.t_step += 1



    def learn(self):
        """Trains the Deep QNetwork and returns action values.
           Can use multiple frameworks.
        """
        batch = self.buffer.sample()

        state_batch = torch.cat(batch.state) #[64,1]
        action_batch = torch.cat(batch.action) #[64,1]
        reward_batch = torch.cat(batch.reward) #[64,1]
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device, dtype=torch.uint8)
        next_states = torch.cat([s for s in batch.next_state if s is not None])

        qhat_next_values = torch.zeros(self.batchsize, device=self.device) #[64]
        if self.framework == 'DQN':
            #VANILLA DQN: get max predicted Q values for the next states from the target model
            qhat_next_values[non_final_mask] = self.qhat(next_states).detach().max(1)[0] #[64]

        if self.framework == 'D2DQN':
            #DOUBLE DQN: get maximizing action under Q, evaluate actionvalue under qHAT
            q_next_actions = torch.zeros(self.batchsize, device=self.device, dtype=torch.long) #[64]
            q_next_actions[non_final_mask] = self.q(next_states).detach().argmax(1) #[64]
            qhat_next_values[non_final_mask] = self.qhat(next_states).gather(1, q_next_actions.unsqueeze(1)).squeeze(1) #[64]

        values = self.q(state_batch).gather(1, action_batch) #[64,1]
        expected_values = reward_batch + (self.gamma * qhat_next_values) #[64]

        #Huber Loss provides better results than MSE
        loss = F.smooth_l1_loss(values, expected_values.unsqueeze(1)) #[64,1]
        #backpropogate
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.q.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def update_epsilon(self):
        self.epsilon = max(self.epsilon*self.epsilon_decay, self.epsilon_min)

    def _set_optimizer(self, params):
        if self.optimizer == "RMSprop":
            return optim.RMSprop(params, lr=self.lr, momentum=self.momentum)
        elif self.optimizer == "SGD":
            return optim.SGD(params, lr=self.lr, momentum=self.momentum)
        else:
            return optim.Adam(params, lr=self.lr)

    def _make_model(self, use_cnn):
        if use_cnn:
            print("Using Pixel-based training.")
            return QCNNetwork(self.nS, self.nA, self.seed).to(self.device)
        else:
            print("Using state data provided by the engine for training.")
            return QNetwork(self.nS, self.nA, self.seed, self.dropout).to(self.device)




class ReplayBuffer:
    def __init__(self, buffersize, batchsize, framestack, device):
        self.buffer = deque(maxlen=buffersize)
        self.stack = deque(maxlen=framestack)
        self.batchsize = batchsize
        self.memory = namedtuple("memory", field_names=['state','action','reward','next_state'])
        self.device = device

    def stack(self, state):
        return

    def add(self, state, action, reward, next_state):
        t = self.memory(state, action, reward, next_state)
        self.buffer.append(t)

    def sample(self):
        batch = random.sample(self.buffer, k=self.batchsize)
        return  self.memory(*zip(*batch))

    def __len__(self):
        return len(self.buffer)




class QCNNetwork(nn.Module):
    """Deep Q-Network CNN Model for use with learning from pixel data.
       Nonlinear estimator for Qπ
    """

    def __init__(self, state, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QCNNetwork, self).__init__()
        chans, width, height = state

        outs = [32, 64, 64]
        kernels = [8, 4, 3]
        strides = [4, 2, 1]
        fc_hidden = 512

        self.conv1 = nn.Conv2d(chans, outs[0], kernels[0], stride=strides[0])
        self.conv2 = nn.Conv2d(outs[0], outs[1], kernels[1], stride=strides[1])
        self.conv3 = nn.Conv2d(outs[1], outs[2], kernels[2], stride=strides[2])

        fc = np.array([width, height])
        for _, kernel, stride in zip(outs, kernels, strides):
            fc = (fc - (kernel - 1) - 1) // stride  + 1
        fc_in = outs[-1] * fc[0] * fc[1]

        self.fc1 = nn.Linear(fc_in, fc_hidden)
        self.fc2 = nn.Linear(fc_hidden, action_size)
        self.seed = torch.manual_seed(seed)


    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x



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

        self.seed = torch.manual_seed(seed)
        self.hidden_layers = nn.ModuleList([nn.Linear(state_size, layer_sizes[0])])
        self.output = nn.Linear(layer_sizes[-1], action_size)

        layer_sizes = zip(layer_sizes[:-1], layer_sizes[1:])
        self.hidden_layers.extend([nn.Linear(i, o) for i, o in layer_sizes])
        self.dropout = nn.Dropout(p=dropout)


    def forward(self, state):
        """Build a network that maps state -> action values."""


        x = F.relu(self.hidden_layers[0](state))
        x = self.dropout(x)
        for layer in self.hidden_layers[1:]:
            x = F.relu(layer(x))
            x = self.dropout(x)
        return self.output(x)
