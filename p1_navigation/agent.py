import numpy as np
import random
from collections import namedtuple, deque

from model import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

class DQN_Agent():
    """Uses a classic Deep Q-Network to learn from the environment"""
