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


n = deque(maxlen=5)

n.append("a")
n.append("b")
n.append("c")
n.append("d")
n.append("e")

print(n)


n.append("f")
print(n)


for letter in n:
    print(letter)

n.appendleft("g")
for letter in n:
    print(letter)
