from IPython.display import Image, display
import os
import os.path
import matplotlib.pyplot as plt
from PIL import Image

def print_args(args):
    print('\n'.join(["{}: {}".format(arg, getattr(args, arg)) for arg in vars(args)]))

def plot_results(imdir):
    images = [os.path.join(imdir, file) for file in os.listdir(imdir) if os.path.splitext(file)[1] == '.png']
    num = len(images)
    for img in images:
        display(Image.open(img))

def print_env_info(state, action, reward):
    print("The agent chooses ACTIONS that look like:\n{}\n".format(action))
    print("The environment returns STATES that look like:\n{}\n".format(state))
    print("The environment returns REWARDS that look like:\n{}".format(reward))
