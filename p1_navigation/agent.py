import numpy as np
import random
from collections import namedtuple, deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# import torchvision.transforms as T

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
        self.gamma = args.gamma
        self.lr = args.learn_rate
        self.update_every = args.update_every
        self.momentum = args.momentum
        self.no_per = args.no_prioritized_replay
        self.train = args.train
        self.pixels = args.pixels

        #Initialize Q-Network
        self.q = self._make_model(args.pixels)
        self.qhat = self._make_model(args.pixels)
        self.qhat.load_state_dict(self.q.state_dict())
        self.qhat.eval()
        self.optimizer = self._set_optimizer(self.q.parameters())

        #initialize REPLAY buffer
        if self.no_per:
            self.memory = ReplayBuffer(self.buffersize, self.batchsize, self.framestack, self.device, self.nS, self.pixels)
        else:
            self.memory = PERBuffer(self.buffersize, self.batchsize, self.framestack, self.device, args.alpha, args.beta)
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
            action = action_values.max(1)[1].view(1,1)
            #print("Using greedy action:", action.item())
            return action
        else:
            #print("Using random action.")
            return torch.tensor([[random.randrange(self.nA)]], device=self.device, dtype=torch.long)

    def step(self, state, action, reward, next_state):
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

    def _memory_loaded(self):
        """Determines whether it's safe to start learning because the memory is
           sufficiently filled.
        """
        # if self.memory.type == "ReplayBuffer":
        #     pass
        return

    def learn(self):
        """Trains the Deep QNetwork and returns action values.
           Can use multiple frameworks.
        """
        #If using standard ReplayBuffer, is_weights & tree_idx will return None
        batch, is_weights, tree_idx = self.memory.sample(self.batchsize)

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
            q_next_actions = self.q(next_states).detach().argmax(1)
            qhat_next_values[non_final_mask] = self.qhat(next_states).gather(1, q_next_actions.unsqueeze(1)).squeeze(1)

        expected_values = reward_batch + (self.gamma * qhat_next_values) #[64]

        expected_values = expected_values.unsqueeze(1) #[64,1]
        values = self.q(state_batch).gather(1, action_batch) #[64,1]


        if is_weights is None:
            #Huber Loss provides better results than MSE
            loss = F.smooth_l1_loss(values, expected_values) #[64,1]
        else:
            #Compute Huber Loss manually to utilize is_weights
            loss, td_errors = self.criterion.huber(values, expected_values, is_weights)
            self.memory.batch_update(tree_idx, td_errors)

        #backpropogate
        self.optimizer.zero_grad()
        loss.backward()
        # for param in self.q.parameters():
        #     param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def update_epsilon(self, ed, em):
        self.epsilon = max(self.epsilon * ed, em)

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
        #print(len(state))
        chans, width, height = state

        # outs = [32, 64, 64]
        # kernels = [8, 4, 3]
        # strides = [4, 2, 1]
        outs = [128, 128, 128]
        kernels = [5, 5, 5]
        strides = [2, 2, 2]

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
    """
    Standard replay buffer to hold memories for use in a DQN Agent. Returns
    random experiences with no consideration of their usefulness.
    """
    def __init__(self, buffersize, batchsize, framestack, device, nS, pixels):
        self.buffer = deque(maxlen=buffersize)
        self.batchsize = batchsize
        self.memory = namedtuple("memory", field_names=['state','action','reward','next_state'])
        self.device = device
        self.framestack = framestack
        self.type = "ReplayBuffer"
        print("Using standard random Replay memory buffer.")

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
    Use a SumTree data structure to efficiently catalogue replay experiences.
    The Prioritized Experience Replay buffer will return batches based on how
    much learning is estimated to be possible from each memory.
    """
    priority_epsilon = 0.01  # Ensure no experiences end up with a 0-prob of being sampled
    beta_incremement = 0.001
    max_error = 1.0  # clipped abs error

    def __init__(self, capacity, batchsize, framestack, device, alpha, beta):
        self.tree = SumTree(capacity) # Making the tree
        self.framestack = framestack
        self.device = device
        self.memory = namedtuple("memory", field_names=['state','action','reward','next_state'])
        self.alpha = alpha
        self.beta = beta
        print("Using Priorized Experience Replay memory buffer.")

    def _leaf_values(self):
        """
        Return the values of all the leaves in the SumTree.
        """
        return self.tree.tree[self.tree.branches:]

    def store(self, state, action, reward, next_state):
        """
        Store a new experience in our tree, new experiences are stored with
        maximum priority until they have been replayed at least once in order
        to ensure learning.
        """
        max_priority = np.max(self._leaf_values())
        experience = self.memory(state, action, reward, next_state)
        if max_priority == 0:
            max_priority = self.max_error

        self.tree.add(max_priority, experience)   # set the max p for new p

    def sample(self, n):
        """
        - First, to sample a minibatch of k size, the range [0, priority_total] is / into k ranges.
        - Then a value is uniformly sampled from each range
        - We search in the sumtree, the experience where priority score correspond to sample values are retrieved from.
        - Then, we calculate IS weights for each minibatch element
        """
        batch = []
        idxs = []
        priorities = []
        segment_size = self.tree.total_priority / n

        #idxs = np.empty((n,), dtype=np.int32)
        #is_weights = np.empty((n, 1), dtype=np.float32)


        self.beta = np.min([self.beta + self.beta_incremement, 1.0])  # anneal Beta to 1

        # Calculating the max_weight
        # p_min = np.min(self._leaf_values()) / self.tree.total_priority
        # max_weight = np.power(n * p_min, -self.beta)


        for i in range(n):
            low = segment_size * i
            high = segment_size * (i + 1)

            value = np.random.uniform(low, high) #sample a value from the segment

            index, priority, data = self.tree.get(value) #retrieve corresponding experience
            idxs.append(index)
            priorities.append(priority)
            batch.append(data)
            # p = priority / self.tree.total_priority
            # is_weights[i, 0] = np.power(n * p, -self.beta) / max_weight
            # indexes[i]= index

        probabilities = priorities / self.tree.total_priority
        #Dividing by the max of the batch is a bit different than the original
        #paper, but should accomplish the same goal of always scaling downwards
        #but should be slightly faster than calculating on the entire tree
        is_weights = np.power(self.tree.num_entries * probabilities, -self.beta)
        is_weights /= is_weights.max()
        is_weights = torch.from_numpy(is_weights).type(torch.FloatTensor).to(self.device)
        return self.memory(*zip(*batch)), is_weights, idxs

    """
    Update the priorities on the tree
    """
    def batch_update(self, tree_idx, td_errors):
        #error should be provided to this function as ABS()
        td_errors += self.priority_epsilon  # Add small epsilon to error to avoid ~0 probability
        clipped_errors = np.minimum(td_errors, self.max_error) # No error should be weight more than a pre-set maximum value
        priorities = np.power(clipped_errors, self.alpha) # Raise the TD-error to power of Alpha to tune between fully weighted or fully random

        for idx, priority in zip(tree_idx, priorities):
            self.tree.update(idx, priority)

    def __len__(self):
        return self.tree.tree_size



class SumTree(object):
    """
    SumTree for holding Prioritized Replay information in an efficiently to
    sample data structure. Uses While-loops throughought instead of commonly
    used recursive function calls, for small efficiency gain.
    """
    def __init__(self, capacity):
        self.capacity = capacity # Number of memories to store
        self.branches = capacity - 1 # Branches above the leafs that store sums
        self.tree_size = self.capacity + self.branches # Total tree size
        self.tree = np.zeros(self.tree_size) # Create SumTree array
        self.data = np.zeros(capacity, dtype=object)  # Create array to hold memories corresponding to SumTree leaves
        self.data_pointer = 0
        self.num_entries = 0

    def add(self, priority, data):
        """
        Add the memory in DATA
        Add priority score in the TREE leaf
        """
        idx = self.data_pointer % self.capacity
        tree_index = self.branches + idx # Update the leaf

        self.data[idx] = data # Update data frame
        self.update(tree_index, priority) # Indexes are added sequentially
        # Incremement
        self.data_pointer += 1
        if self.num_entries < self.capacity:
            self.num_entries += 1

    def update(self, tree_index, new_priority):
        """
        Update the leaf priority score and propagate the change through tree
        """
        # Change = new priority score - former priority score
        change = new_priority - self.tree[tree_index]
        self.tree[tree_index] = new_priority
        # then propagate the change through tree
        while tree_index != 0:
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change

    def get(self, v):
        current_idx = 0
        while True:
            left = 2 * current_idx + 1
            right = left + 1

            # If we reach bottom, end the search
            if left >= self.tree_size:
                leaf_index = current_idx
                break
            else: # downward search, always search for a higher priority node
                if v <= self.tree[left]:
                    current_idx = left

                else:
                    v -= self.tree[left]
                    current_idx = right

        data_index = leaf_index - self.capacity + 1

        return leaf_index, self.tree[leaf_index], self.data[data_index]

    @property
    def total_priority(self):
        return self.tree[0] # Returns the root node
