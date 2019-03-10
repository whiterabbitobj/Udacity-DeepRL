import time
import numpy as np
import torch
import progressbar

from utils import print_bracketing, LogLoader
from PIL import Image
import torchvision.transforms as T
from collections import namedtuple, deque


l = LogLoader()

f = "C:\\Users\\mattdoll\\Development\\deep-reinforcement-learning\\p2_continuous-control\\saves\\D4PG_20190309_v10\\logs\\D4PG_20190309_v10_LOG.txt"
print(f)

l.list_params(f)



l.load_params(f)
