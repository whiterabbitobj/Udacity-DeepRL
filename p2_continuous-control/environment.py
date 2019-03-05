import os.path
import time
import re

import torch
import matplotlib.pyplot as plt
import numpy as np
from unityagents import UnityEnvironment
from collections import deque
# import torchvision.transforms as T


class Environment:
    """
    Wrapper for the Udacity Reacher environment utilizing 20 actors. Keeps the
    main body code a bit more neat and allows for easier access to certain
    params elsewhere.
    """
    def __init__(self, args):

        self.train = args.train

        self.env = UnityEnvironment(file_name='Reacher_Windows_x86_64/Reacher.exe', no_graphics=args.nographics)
        self.brain_name = self.env.brain_names[0]
        self.brain = self.env.brains[self.brain_name]

        # Environment resets itself when the class is instantiated
        self.reset()

        self.action_size = self.brain.vector_action_space_size
        self.state_size = self.states.shape[1]
        self.agent_count = len(self.env_info.agents)

    def reset(self):
        """
        Resets the environment.
        """
        self.env_info = self.env.reset(train_mode=self.train)[self.brain_name]

    def close(self):
        """
        Closes the environment when Agent is done interacting with it.
        """
        self.env.close()

    def step(self, actions):
        """
        Returns REWARDS, NEXT_STATES, DONES based on the actions provided.
        """
        self.env_info = self.env.step(actions)[self.brain_name]
        next_states = self.states
        rewards = self.env_info.rewards
        dones = self.env_info.local_done
        return next_states, rewards, dones

    @property
    def states(self):
        """
        Returns the STATES as a tensor.
        """
        states = self.env_info.vector_observations
        #states = self._normalize_states(states)
        return torch.from_numpy(states).float()