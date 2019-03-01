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
for i in range(4):
    print("hello world")
    if i == 2:
        break

5e4



5**0


class ReplayBuffer:
    def __init__(self):
        pass
    def print(self, statement):
        print(statement)


x = ReplayBuffer()
print(x)
x.print("howdy")
print(x.__class__.__name__)


spinner = Spinner("Loading ")
for _ in range(20):
    spinner.next()
    time.sleep(0.1)
