import numpy as np
import random
import torch

from collections import namedtuple, deque


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
