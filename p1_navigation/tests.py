import time
import numpy as np
import torch
import progressbar

# from unityagents import UnityEnvironment
from agent import Agent
import utils
from get_args import get_args
from PIL import Image
import torchvision.transforms as T
from collections import namedtuple, deque


class ReplayBuffer:
    def __init__(self, buffersize, batchsize, framestack, device, nS, pixels):
        self.buffer = deque(maxlen=buffersize)
        self.batchsize = batchsize
        self.memory = namedtuple("memory", field_names=['state','action','reward','next_state'])
        self.device = device
        self.framestack = framestack
        self.type = "ReplayBuffer"
        print("Using standard stochastic Replay memory buffer.")

    def store(self, state, action, reward, next_state):
        t = self.memory(state, action, reward, next_state)
        self.buffer.append(t)

    def sample(self):
        batch = random.sample(self.buffer, k=self.batchsize)
        return  self.memory(*zip(*batch)), None, None

    def __len__(self):
        return len(self.buffer)


x = ReplayBuffer(1000, 64, 4, "cpu", 6, True)
print(type(x))


print(2%4)
