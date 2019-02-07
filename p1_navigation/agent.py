import numpy as np
import random
from collections import namedtuple, deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Agent():
    """Uses a classic Deep Q-Network to learn from the environment"""

    def __init__(self, nS, nA, args, seed=0):
        #super(DQN_Agent, self).__init()

        #initialize agent parameters
        self.nS = nS
        self.nA = nA
        self.seed = random.seed(seed)
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

        #initialize REPLAY buffer
        self.buffer = ReplayBuffer(self.buffersize, self.batchsize, self.device)

        #Initialize Q-Network
        self.q = _make_model(args.pixels)
        self.qhat = _make_model(args.pixels)
        self.qhat.load_state_dict(self.q.state_dict())
        self.optimizer = _set_optimizer(self.q.parameters())

    def act(self, state):
        #send the state to a tensor object on the gpu
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)

        self.q.eval() #put network into eval mode
        with torch.no_grad():
            action_values = self.q(state)
        self.q.train() #put network back into training mode
        #select an action using epsilon-greedy π
        if random.random() > self.epsilon or not self.train:
            return np.argmax(action_values.cpu().data.numpy()).astype(int)
        else:
            return random.choice(np.arange(self.nA))

    def step(self, state, action, reward, next_state, done):
        """Moves the agent to the next timestep.
           Learns each UPDATE_EVERY steps.
        """
        #save the current SARS′  status in the replay buffer
        self.buffer.add(state, action, reward, next_state, done)

        if len(self.buffer) > self.batchsize and self.t_step % self.update_every == 0:
            batch = self.buffer.sample()
            self.learn(batch)
        #update the target network every C steps
        if self.t_step % self.C == 0:
            self.qhat.load_state_dict(self.q.state_dict())
        self.t_step += 1



    def learn(self, batch):
        """Trains the Deep QNetwork and returns action values.
           Can use multiple frameworks.
        """
        states, actions, rewards, next_states, dones = batch

        if self.framework == 'DQN':
            #VANILLA DQN: get max predicted Q values for the next states from the target model
            qhat_nextvalues = self.qhat(next_states).detach().max(1)[0].unsqueeze(1)

        if self.framework == 'D2DQN':
            #DOUBLE DQN: get maximizing action under Q, evaluate actionvalue under qHAT
            q_next_actions = self.q(next_states).detach().argmax(1).unsqueeze(1)
            qhat_nextvalues = self.qhat(next_states).gather(1, q_next_actions)

        expected_values = rewards + (self.gamma * qhat_nextvalues * (1 - dones))
        values = self.q(states).gather(1, actions)

        #loss = F.mse_loss(values, expected_values)
        loss = F.smooth_l1_loss(values, expected_values)

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
            return QCNNetwork(nS, nA, seed).to(self.device)
        else:
            print("Using state data provided by the engine for training.")
            return QNetwork(nS, nA, seed, self.dropout).to(self.device)




class ReplayBuffer:
    def __init__(self,buffersize, batchsize, device):
        self.buffer = deque(maxlen=buffersize)
        self.batchsize = batchsize
        self.memory = namedtuple("memory", field_names=['state','action','reward','next_state','done'])
        self.device = device

    def add(self, state, action, reward, next_state, done):
        t = self.memory(state, action, reward, next_state, done)
        self.buffer.append(t)

    def sample(self):
        batch = random.sample(self.buffer, k=self.batchsize)

        states = torch.from_numpy(np.vstack([np.expand_dims(memory.state, axis=0) for memory in batch if memory is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([memory.action for memory in batch if memory is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([memory.reward for memory in batch if memory is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([np.expand_dims(memory.next_state, axis=0) for memory in batch if memory is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([memory.done for memory in batch if memory is not None]).astype(np.uint8)).float().to(self.device)

        return (states, actions, rewards, next_states, dones)

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

        self.conv1 = nn.Conv2d(chans, outs[0], kernels[0], stride=stride[0])
        self.conv2 = nn.Conv2d(out[0], outs[1], kernel[1], stride=stride[1])
        self.conv3 = nn.Conv2d(out[1], out[2], kernel[2], stride=stride[2])

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
