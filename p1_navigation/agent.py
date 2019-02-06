import numpy as np
import random
from collections import namedtuple, deque

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
        self.t_step = 0

        #initialize params from the command line args
        self.framework = args.framework
        self.batchsize = args.batchsize
        self.buffersize = args.buffersize
        self.c = args.c
        self.dropout = args.dropout
        self.epsilon = args.epsilon
        self.epsilon_decay = args.epsilon_decay
        self.epsilon_min = args.epsilon_min
        self.gamma = args.gamma
        self.lr = args.learn_rate
        self.tau = args.tau
        self.update_every = args.update_every
        self.momentum = args.momentum
        self.PER = args.prioritized_replay

        #initialize REPLAY buffer
        self.buffer = ReplayBuffer(nA, self.buffersize, self.batchsize, seed, device)

        #Initialize a Q-Network
        if not args.pixels:
            print("Using api based training using state data provided by the engine.")
            self.q = QNetwork(nS, nA, seed, self.dropout).to(device)
            self.qhat = QNetwork(nS, nA, seed, self.dropout).to(device)
        else:
            print("Using Pixel-based training.")
            self.q = QCNNetwork(nS, nA, seed, self.dropout).to(device)
            self.qhat = QCNNetwork(nS, nA, seed, self.dropout).to(device)

        self.qhat.load_state_dict(self.q.state_dict())

        #set optimizer
        if args.optimizer == "RMSprop":
            self.optimizer = optim.RMSprop(self.q.parameters(), lr=self.lr, momentum=self.momentum)
        elif args.optimizer == "Adam":
            self.optimizer = optim.Adam(self.q.parameters(), lr=self.lr)
        else:
            self.optimizer = optim.SGD(self.q.parameters(), lr=self.lr, momentum=self.momentum)

    def update_epsilon(self):
        self.epsilon = max(self.epsilon*self.epsilon_decay, self.epsilon_min)

    def act(self, state):
        #send the state to a tensor object on the gpu
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.q.eval() #put network into eval mode
        with torch.no_grad():
            action_values = self.q(state)
        self.q.train() #put network back into training mode
        #select an action using epsilon-greedy π
        if random.random() > self.epsilon:
            return np.argmax(action_values.cpu().data.numpy()).astype(int)
        else:
            return random.choice(np.arange(self.nA))

    def step(self, state, action, reward, next_state, done):
        """Moves the agent to the next timestep.
           Learns every UPDATE_EVERY steps.
        """
        #save the current SARS′  status in the replay buffer
        self.buffer.add(state, action, reward, next_state, done)

        if len(self.buffer) > self.batchsize and self.t_step % self.update_every == 0:
            batch = self.buffer.sample(self.PER)
            self.learn(batch)
        #update the target network every C steps
        if self.t_step % self.c == 0:
            self.qhat.load_state_dict(self.q.state_dict())
        self.t_step += 1

    def teststep(self, state, action, reward, next_state, done):
        """Moves the agent to the next timestep.
           Learns every UPDATE_EVERY steps.
        """
        batch = state, action, reward, next_state, done
        self.learn(batch)
        #update the target network every C steps
        if self.t_step % self.c == 0:
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

        loss = F.mse_loss(values, expected_values)

        #minimize the loss (backpropogation)
        self.optimizer.zero_grad()
        loss.backward()
        # for param in self.q.parameters():
        #     param.grad.data.clamp(-1,1)
        self.optimizer.step()



class ReplayBuffer:
    def __init__(self, nA, buffersize, batchsize, seed, device):
        self.nA = nA
        self.buffer = deque(maxlen=buffersize)
        self.batchsize = batchsize
        self.memory = namedtuple("memory", field_names=['state','action','reward','next_state','done'])
        self.seed = random.seed(seed)
        self.device = device

    def add(self, state, action, reward, next_state, done):
        t = self.memory(state, action, reward, next_state, done)
        self.buffer.append(t)

    def sample(self, per):
        batch = random.sample(self.buffer, k=self.batchsize)

        states = torch.from_numpy(np.vstack([memory.state for memory in batch if memory is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([memory.action for memory in batch if memory is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([memory.reward for memory in batch if memory is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([memory.next_state for memory in batch if memory is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([memory.done for memory in batch if memory is not None]).astype(np.uint8)).float().to(self.device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.buffer)




class QCNNetwork(nn.Module):
    """Deep Q-Network CNN Model for use with learning from pixel data.
       Nonlinear estimator for Qπ
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
        super(QCNNetwork, self).__init__()

        # 1 input image channel (grayscale), 10 output channels/feature maps
        # 3x3 square convolution kernel
        ## output size = (W-F)/S +1 = (28-3)/1 +1 = 26
        # the output Tensor for one image, will have the dimensions: (10, 26, 26)
        # after one pool layer, this becomes (10, 13, 13)
        # in_rez = 84
        # chan_count = 3
        # out1 = 32
        # kernel1 = 8
        # stride1 = 4
        #
        # out2 = 64
        # kernel2 = 4
        # stride2 = 2
        #
        # out3 =
        #
        # pool_stride = 2
        # fc_handoff = 64
        #
        # self.conv1 = nn.Conv2d(chan_count, out1, kernel1, stride1, (kernel1-1)/2)
        # self.conv2 = nn.Conv2d(out1, out2, kernel2, stride1, (kernel2-1)/2)
        #
        # self.pool = nn.MaxPool2d(2, pool_stride)
        #
        # conv1_out =  (in_rez-kernel1)/stride1 + 1) / pool_stride
        # conv2_out =  ((conv1_out-kernel2)/stride1 + 1) / pool_stride
        # fc_in = out2 * conv2_out * conv2_out
        #
        #
        # self.fc1 = nn.Linear(fc_in, fc_handoff)
        # self.fc2 = nn.Linear(fc_handoff, action_size)
        #
        # self.dropout = nn.Dropout(p=dropout)
        # self.seed = torch.manual_seed(seed)

        in_rez = 84
        chan_count = 3

        out1 = 32
        kernel1 = 8
        stride1 = 4

        out2 = 64
        kernel2 = 4
        stride2 = 2

        out3 = 64
        kernel3 = 3
        stride3 = 1

        pool_stride = 2

        self.conv1 = nn.Conv2d(chan_count, out1, kernel1, stride=stride1, padding=int((kernel1-1)/2))
        self.conv2 = nn.Conv2d(out1, out2, kernel2, stride=stride2, padding=int((kernel2-1)/2))
        self.conv3 = nn.Conv2d(out2, out3, kernel3, stride=stride3, padding=int((kernel3-1)/2))


        self.pool = nn.MaxPool2d(2, pool_stride)

        # conv1_out = (in_rez-kernel1)/stride1 + 1) / pool_stride
        # conv2_out = ((conv1_out-kernel2)/stride2 + 1) / pool_stride
        # conv3_out = ((conv2_out-kernel3)/stride3 + 1) / pool_stride
        # fc_in = out2 * conv2_out * conv2_out
        fc_in = 512
        fc_handoff = 512

        self.fc1 = nn.Linear(fc_in, fc_handoff)
        self.fc2 = nn.Linear(fc_handoff, action_size)

        self.seed = torch.manual_seed(seed)


    def forward(self, state):
        """Build a network that maps state -> action values."""
        #
        # x = self.pool(F.relu(self.conv1(state)))
        # x = self.pool(F.relu(self.conv2(x)))
        #
        # x = x.view(x.size(0), -1)
        #
        # x = F.relu(self.fc1(x))
        # x = self.dropout(x)
        # x = self.fc2(x)
        #
        # return x

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


        x = F.relu(self.hidden_layers[0](state))
        x = self.dropout(x)
        for layer in self.hidden_layers[1:]:
            x = F.relu(layer(x))
            x = self.dropout(x)
        return self.output(x)
