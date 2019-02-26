import os.path
import time
import re

import torch
import matplotlib.pyplot as plt
import numpy as np
# from agent import Agent
from unityagents import UnityEnvironment
#from PIL import Image
from collections import deque
import torchvision.transforms as T


class Environment:


    def __init__(self, args):

        self.train = args.train

        self.env = UnityEnvironment(file_name='Reacher_Windows_x86_64/Reacher.exe', no_graphics=args.nographics)
        self.brain_name = self.env.brain_names[0]
        self.brain = self.env.brains[self.brain_name]

        self.reset()

        self.action_size = self.brain.vector_action_space_size
        self.state_size = self.states.shape[1]
        self.agent_count = len(self.env_info.agents)

    def reset(self):
        self.env_info = self.env.reset(train_mode=self.train)[self.brain_name]

    def close(self):
        self.env.close()

    def step(self, actions):
        self.env_info = self.env.step(actions)[self.brain_name]
        rewards = self.env_info.rewards
        dones = self.env_info.local_done
        return rewards, dones


    @property
    def states(self):
        return self.env_info.vector_observations
