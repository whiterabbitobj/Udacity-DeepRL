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


class Test:
    def __init__(self):
        self.time = self.atime = time.time()

    def newTime(self):
        self.atime = time.time()

t = Test()
print(t.time, t.atime)
t.newTime()
print(t.time, t.atime)
