from collections import deque
import random
import torch
import numpy as np

class ReplayBuffer:
    """
    Standard replay buffer to hold memories for later learning. Returns
    random experiences with no consideration of their usefulness.

    When using an agent with a ROLLOUT trajectory, then instead of each
    experience holding SARS' data, it holds:
    state = state at t
    action = action at t
    reward = cumulative reward from t through t+n-1
    next_state = state at t+n

    where n=ROLLOUT.
    """
    def __init__(self, device, buffer_size=100000, gamma=0.99, rollout=5):
        self.buffer = deque(maxlen=buffer_size)
        self.device = device
        self.gamma = gamma
        self.rollout = rollout

    def store_trajectory(self, state, action, reward, next_state):
        """
        Stores a trajectory, which may or may not be the same as an experience,
        but allows for n_step rollout.
        """

        trajectory = (state, action, reward, next_state)
        self.buffer.append(trajectory)

    def sample(self, batch_size):
        """
        Return a sample of size BATCH_SIZE as a tuple.
        """
        batch = random.sample(self.buffer, k=batch_size)
        states, actions, rewards, next_states = zip(*batch)

        states = torch.cat(states).to(self.device)
        actions = torch.cat(actions).to(self.device).long()
        rewards = torch.cat(rewards).to(self.device)
        terminal_mask = torch.tensor(tuple(map(lambda s: s is not None, next_states)), dtype=torch.uint8).to(self.device)
        next_states = torch.cat([s for s in next_states if s is not None]).to(self.device)

        batch = (states, actions, rewards, next_states, terminal_mask)
        return batch, None, None

    def init_n_step(self):
        """
        Creates (or recreates to zero an existing) deque to handle nstep returns.
        """
        self.n_step = deque(maxlen=self.rollout)

    def store_experience(self, experience):
        """
        Once the n_step memory holds ROLLOUT number of sars' tuples, then a full
        memory can be added to the ReplayBuffer.
        """
        self.n_step.append(experience)
        # Abort if ROLLOUT steps haven't been taken in a new episode
        if len(self.n_step) < self.rollout:
            return

        # Unpacks and stores the SARS' tuple for each actor in the environment
        # for actor in zip(*self.n_step):
        states, actions, rewards, next_states = zip(*self.n_step)
        n_steps = self.rollout - 1


        # Calculate n-step discounted reward
        # If encountering a terminal state (next_state == None) then sum the
        # rewards only until the terminal state and report back a terminal state
        # for the experience tuple.
        if n_steps > 0:
            reward = torch.tensor(0, dtype=torch.float)
            for i in range(n_steps):
                if next_states[i] is  None:
                    print("Encountered terminal state in ROLLOUT stacking! Reward calculated from {} steps: {}".format(i+1, reward))
                    n_steps = i
                    break
                else:
                    reward += self.gamma**i * rewards[i]

            rewards = reward

            #rewards = np.fromiter((self.gamma**i * rewards[i] for i in range(n_steps)), float, count=n_steps)
            #rewards = rewards.sum()

        # store the current state, current action, cumulative discounted
        # reward from t -> t+n-1, and the next_state at t+n (S't+n)
        state = states[0]
        action = torch.from_numpy(actions[0])
        reward = torch.tensor([rewards])
        next_state = next_states[n_steps]
        self.store_trajectory(state, action, reward, next_state)

    def __len__(self):
        return len(self.buffer)
#
# class ReplayBuffer:
#     """
#     Standard replay buffer to hold memories for use in a DQN Agent. Returns
#     random experiences with no consideration of their usefulness.
#     """
#     def __init__(self, buffersize, framestack, device, nS, pixels):
#         self.buffer = deque(maxlen=buffersize)
#         self.memory = namedtuple("memory", field_names=['state','action','reward','next_state'])
#         self.device = device
#         self.framestack = framestack
#         self.type = "ReplayBuffer"
#         print("Using standard random Replay memory buffer.")
#
#     def store(self, state, action, reward, next_state):
#         t = self.memory(state, action, reward, next_state)
#         self.buffer.append(t)
#
#     def sample(self, batchsize):
#         batch = random.sample(self.buffer, k=batchsize)
#         return  self.memory(*zip(*batch)), None, None
#
#     def __len__(self):
#         return len(self.buffer)



class PERBuffer(object):  # stored as ( s, a, r, s_ ) in SumTree
    """
    Use a SumTree data structure to efficiently catalogue replay experiences.
    The Prioritized Experience Replay buffer will return batches based on how
    much learning is estimated to be possible from each memory.
    """
    priority_epsilon = 0.01  # Ensure no experiences end up with a 0-prob of being sampled
    beta_incremement = 0.00001
    max_error = 1.0  # clipped abs error

    def __init__(self, capacity, batchsize, framestack, device, alpha, beta):
        self.tree = SumTree(capacity) # Making the tree
        self.framestack = framestack
        self.device = device
        self.memory = namedtuple("memory", field_names=['state','action','reward','next_state'])
        self.alpha = alpha
        self.beta = beta
        self.type = "PERBuffer"
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
        Sample experiences from the replay buffer using stochastic prioritization
        based on TD-error. Returns a batch evenly distributed in N samples.
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

    def batch_update(self, tree_idx, td_errors):
        """
        Update the priorities on the tree
        """
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
        """
        Retrieve the index, values at that index, and replay memory, that is
        most closely associated to sample value of V.
        """
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
