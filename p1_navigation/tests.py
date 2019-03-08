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



l = [1,2,3]
k = [4,5,6]
l += k
print(l)
