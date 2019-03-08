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
from progress.bar import Bar
from progress.spinner import Spinner



x = np.array(([1,2,3],[4,5,6]))
y = np.array(([.1,.2,.3],[.4,.5,.6]))
print(x.shape)
print(y.shape)

a = (x * y).sum(-1)
print(a)
b = x.sum(-1) + y.sum(-1)
print(b)
