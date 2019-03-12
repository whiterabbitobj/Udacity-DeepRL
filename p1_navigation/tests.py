import time
import numpy as np
import torch
import progressbar

# from unityagents import UnityEnvironment
from agent import Agent
import utils
from utils import LogLoader
from get_args import get_args
from PIL import Image
import torchvision.transforms as T
from collections import namedtuple, deque
from progress.bar import Bar
from progress.spinner import Spinner



i = 0
while i < 30:
    i += 1
    if i%5 == 0:
        continue
    print(i)
    
