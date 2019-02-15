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

x = np.min([3, 1])
print(x)


a = [0,1,2,3,4,5,6,7,8]
print(a[-5:])
print(a[4:])
