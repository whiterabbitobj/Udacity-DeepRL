import numpy as np
import random
from collections import namedtuple, deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T

class Agent():
    """Uses a classic Deep Q-Network to learn from the environment"""

    def __init__(self, nA, state_size, args, seed=0):
        #super(DQN_Agent, self).__init()

        #initialize agent parameters
        self.nS = state_size
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
        self.no_per = args.no_prioritized_replay
        self.train = args.train
        self.pixels = args.pixels

        #initialize REPLAY buffer
        if self.no_per:
            self.memory = ReplayBuffer(self.buffersize, self.batchsize, self.framestack, self.device, self.nS, self.pixels)
        else:
            self.memory = PERBuffer(self.buffersize, self.batchsize, self.framestack, self.device, args.alpha, args.beta)

        #Initialize Q-Network
        self.q = self._make_model(args.pixels)
        self.qhat = self._make_model(args.pixels)
        self.qhat.load_state_dict(self.q.state_dict())
        self.qhat.eval()
        self.optimizer = self._set_optimizer(self.q.parameters())
        self.criterion = WeightedLoss()


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
        reward = torch.tensor([reward], device=self.device)
        self.memory.store(state, action, reward, next_state)

        if len(self.memory) >= self.batchsize and self.t_step % self.update_every == 0:
            self.learn()
        #update the target network every C steps
        if self.t_step % self.C == 0:
            self.qhat.load_state_dict(self.q.state_dict())
        self.t_step += 1

    def learn(self):
        """Trains the Deep QNetwork and returns action values.
           Can use multiple frameworks.
        """
        batch, ISWeights, tree_idx = self.memory.sample()

        state_batch = torch.cat(batch.state) #[64,1]
        action_batch = torch.cat(batch.action) #[64,1]
        reward_batch = torch.cat(batch.reward) #[64,1]
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device, dtype=torch.uint8)
        next_states = torch.cat([s for s in batch.next_state if s is not None])

        qhat_next_values = torch.zeros(self.batchsize, device=self.device) #[64]
        if self.framework == 'DQN':
            #VANILLA DQN: get max predicted Q values for the next states from the target model
            qhat_next_values[non_final_mask] = self.qhat(next_states).detach().max(1)[0] #[64]

        if self.framework == 'DDQN':
            #DOUBLE DQN: get maximizing action under Q, evaluate actionvalue under qHAT
            q_next_actions = torch.zeros(self.batchsize, device=self.device, dtype=torch.long) #[64]
            q_next_actions[non_final_mask] = self.q(next_states).detach().argmax(1) #[64]
            qhat_next_values[non_final_mask] = self.qhat(next_states).gather(1, q_next_actions.unsqueeze(1)).squeeze(1) #[64]

        expected_values = reward_batch + (self.gamma * qhat_next_values) #[64]
        expected_values = expected_values.unsqueeze(1) #[64,1]

        values = self.q(state_batch).gather(1, action_batch) #[64,1]


        if ISWeights is None:
            #Huber Loss provides better results than MSE
            loss = F.smooth_l1_loss(values, expected_values) #[64,1]
        else:
            #Compute Huber Loss manually to utilize ISWeights
            loss, td_errors = self.criterion.huber(values, expected_values, ISWeights)
            self.memory.batch_update(tree_idx, td_errors)

<<<<<<< HEAD
        #Huber Loss provides better results than MSE
        loss = F.smooth_l1_loss(values, expected_values.unsqueeze(1)) #[64,1]
        #print("VALUES: {}, EXPECTED: {}, TD_ERRORS: {}".format(values.shape, expected_values.unsqueeze(1).shape, loss.shape))

||||||| merged common ancestors
        #Huber Loss provides better results than MSE
        loss = F.smooth_l1_loss(values, expected_values.unsqueeze(1)) #[64,1]
=======
>>>>>>> class_reorganization
        #backpropogate
        self.optimizer.zero_grad()
        loss.backward()
        # for param in self.q.parameters():
        #     param.grad.data.clamp_(-1, 1)
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
            return QCNNetwork(self.nS, self.nA, self.seed).to(self.device)
        else:
            return QNetwork(self.nS, self.nA, self.seed, self.dropout).to(self.device)



class WeightedLoss(nn.Module):
    def __init__(self):
        super(WeightedLoss, self).__init__()

    def huber(self, values, targets, weights):
        errors = torch.abs(values - targets)
        loss = (errors<1).float()*(errors**2) + (errors>=1).float()*(errors - 0.5)
        weighted_loss = (weights * loss).sum()
        #weighted_loss = weighted_loss.sum()
        return weighted_loss, errors.detach().cpu().numpy()



class QCNNetwork(nn.Module):
    """Deep Q-Network CNN Model for use with learning from pixel data.
       Nonlinear estimator for Qπ
    """
    def __init__(self, state, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state (tensor): Dimension of each state
            action_size (int): Number of possible actions returned from network
            seed (int): Random seed
        """
        super(QCNNetwork, self).__init__()
        chans, width, height = state

        outs = [32, 64, 64]
        kernels = [8, 4, 3]
        strides = [4, 2, 1]
        # outs = [128, 128, 128]
        # kernels = [5, 5, 5]
        # strides = [2, 2, 2]

        self.conv1 = nn.Conv2d(chans, outs[0], kernels[0], stride=strides[0])
        self.bn1 = nn.BatchNorm2d(outs[0])
        self.conv2 = nn.Conv2d(outs[0], outs[1], kernels[1], stride=strides[1])
        self.bn2 = nn.BatchNorm2d(outs[1])
        self.conv3 = nn.Conv2d(outs[1], outs[2], kernels[2], stride=strides[2])
        self.bn3 = nn.BatchNorm2d(outs[2])

        fc_hidden = 512

        fc = np.array([width, height])
        for _, kernel, stride in zip(outs, kernels, strides):
            fc = (fc - (kernel - 1) - 1) // stride  + 1
        fc_in = outs[-1] * fc[0] * fc[1]
        self.fc1 = nn.Linear(fc_in, fc_hidden)
        self.fc2 = nn.Linear(fc_hidden, action_size)
        self.seed = torch.manual_seed(seed)


    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.bn1(self.conv1(state)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
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



class ReplayBuffer:
    def __init__(self, buffersize, batchsize, framestack, device, nS, pixels):
        self.buffer = deque(maxlen=buffersize)
        self.batchsize = batchsize
        self.memory = namedtuple("memory", field_names=['state','action','reward','next_state'])
        self.device = device
        self.framestack = framestack
        print("Using standard stochastic Replay memory buffer.")

    def store(self, state, action, reward, next_state):
        t = self.memory(state, action, reward, next_state)
        self.buffer.append(t)

    def sample(self):
        batch = random.sample(self.buffer, k=self.batchsize)
        return  self.memory(*zip(*batch)), None, None

    def __len__(self):
        return len(self.buffer)



class PERBuffer(object):  # stored as ( s, a, r, s_ ) in SumTree
    """
    This SumTree implementation is a heavily modified version of code from
    Thomas Simonini: https://tinyurl.com/y3y6n2zc
    """
    priority_epsilon = 0.01  # Ensure no experiences end up with a 0-prob of being sampled
    #alpha = 0.6  # Controls likelihood of using high priority or random samples
    #beta = 0.4  # For adjusting the IS weights, beta anneals to 1

    beta_incremement = 0.001

    absolute_error_upper = 1.0  # clipped abs error

    def __init__(self, capacity, batchsize, framestack, device, alpha, beta):
        self.tree = SumTree(capacity) # Making the tree
        self.batchsize = batchsize
        self.framestack = framestack
        self.device = device
        self.memory = namedtuple("memory", field_names=['state','action','reward','next_state'])
        self.alpha = alpha
        self.beta = beta
        print("Using Priorized Experience Replay memory buffer.")

    def _leaf_values(self):
        """
        Return the values of all the leaves in the SumTree
        """
        return self.tree.tree[self.tree.branches:]

    def store(self, state, action, reward, next_state):
        """
        Store a new experience in our tree
        Each new experience have a score of max_prority (it will be then improved when we use this exp to train our DDQN)
        """
        # Find the max priority
        max_priority = np.max(self._leaf_values())
        experience = self.memory(state, action, reward, next_state)
        # If the max priority = 0 we can't put priority = 0 since this exp will never have a chance to be selected
        # So we use a minimum priority
        if max_priority == 0:
            max_priority = self.absolute_error_upper

        self.tree.add(max_priority, experience)   # set the max p for new p

    def sample(self):
        """
        - First, to sample a minibatch of k size, the range [0, priority_total] is / into k ranges.
        - Then a value is uniformly sampled from each range
        - We search in the sumtree, the experience where priority score correspond to sample values are retrieved from.
        - Then, we calculate IS weights for each minibatch element
        """
        n = self.batchsize
        # Create a sample array that will contains the minibatch
        batch = []

        indexes = np.empty((n,), dtype=np.int32)
        ISWeights = np.empty((n, 1), dtype=np.float32)

        # Calculate the size of each segment of the replay memory,
        # based the magnitude of total priority divided into equal segments
        #print(self.tree.total_priority)
        segment_size = self.tree.total_priority / n

        # Here we increasing the beta each time we sample a new minibatch
        self.beta = np.min([self.beta + self.beta_incremement, 1.0])  # max = 1

        # Calculating the max_weight
        p_min = np.min(self._leaf_values()) / self.tree.total_priority
        #p_min = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_priority
        #print(self._leaf_values(), self._leaf_values().shape)
        max_weight = np.power(n * p_min, -self.beta)


        for i in range(n):
            """
            A value is uniformly sample from each range
            """
            low, high = segment_size * i, segment_size * (i + 1)
            value = np.random.uniform(low, high) #sample a value from the segment

            """
            Experience that correspond to each value is retrieved
            """
            index, priority, data = self.tree.get_leaf(value)

            #P(j)
            p = priority / self.tree.total_priority

            #  IS = (N*P(i))**-b  /max wi
            ISWeights[i, 0] = np.power(n * p, -self.beta) / max_weight

            indexes[i]= index

            #experience = [data]
            # batch.append(experience)
            batch.append(data)
        return self.memory(*zip(*batch)), torch.from_numpy(ISWeights).to(self.device), indexes

    """
    Update the priorities on the tree
    """
    def batch_update(self, tree_idx, td_errors):
        #error should be provided to this function as ABS()
        td_errors += self.priority_epsilon  # Add small epsilon to error to avoid ~0 probability
        clipped_errors = np.minimum(td_errors, self.absolute_error_upper) # No error should be weight more than a pre-set maximum value
        priorities = np.power(clipped_errors, self.alpha) # Raise the TD-error to power of Alpha to tune between fully weighted or fully random

        for idx, priority in zip(tree_idx, priorities):
            self.tree.update(idx, priority)

    def __len__(self):
        return self.tree.tree_size



class SumTree(object):
    """
    This SumTree implementation is a heavily modified version of code from
    Thomas Simonini: https://tinyurl.com/y3y6n2zc
    """

    data_pointer = 0

    """
    Here we initialize the tree with all nodes = 0, and initialize the data with all values = 0
    The TREE array (binary tree) holds the priority values
    The DATA array holds the replay memories
    """
    def __init__(self, capacity):
        self.capacity = capacity # Number of memories to store
        self.branches = capacity - 1 # Branches above the leafs that store sums
        self.tree_size = self.capacity + self.branches # Total tree size
        self.tree = np.zeros(self.tree_size) # Create SumTree array
        self.data = np.zeros(capacity, dtype=object)  # Create array to hold memories corresponding to SumTree leaves



    def add(self, priority, data):
        """
        Add the memory in DATA
        Add priority score in the TREE leaf
        """
        idx = self.data_pointer % self.capacity
        self.data[idx] = data # Update data frame
        tree_index = idx + self.branches # Update the leaf
        self.update(tree_index, priority) # Indexes are added sequentially
        # Incremement
        self.data_pointer += 1

    """
    Update the leaf priority score and propagate the change through tree
    """
    def update(self, tree_index, new_priority):
        # Change = new priority score - former priority score
        change = new_priority - self.tree[tree_index]
        self.tree[tree_index] = new_priority
        # then propagate the change through tree
        while tree_index != 0:
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change


    """
    Here we get the leaf_index, priority value of that leaf and experience associated with that index
    """
    def get_leaf(self, v):
        """
        Tree structure and array storage:
        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for experiences
        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        parent_index = 0

        while True:
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1

            # If we reach bottom, end the search
            if left_child_index >= self.tree_size:
                leaf_index = parent_index
                break

            else: # downward search, always search for a higher priority node

                if v <= self.tree[left_child_index]:
                    parent_index = left_child_index

                else:
                    v -= self.tree[left_child_index]
                    parent_index = right_child_index

        data_index = leaf_index - self.capacity + 1

        return leaf_index, self.tree[leaf_index], self.data[data_index]

    @property
    def total_priority(self):
        return self.tree[0] # Returns the root node
