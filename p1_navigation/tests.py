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

def main():

    start_time = time.time()
    args = get_args()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.print_every = np.clip(args.num_episodes//args.print_count, 2, 100)
    args.sep = "#"*50

    env, env_info, brain_name, nA, nS = utils.load_environment(args)
    print("*"*100)

    print("NS:", nS)

    a = Agent(nA, nS, args)
    print(a.buffer.phi)
    s = a.buffer.get_stack()
    print(s, s.shape)

    env.close() #close the environment


if __name__ == "__main__":
    main()



t = (3, 84, 84)
t = list(t)
print(t)
t[0] = 10
print(t)


"""
python banana_deeprl.py --train --pixels --verbose --framework D2DQN -ed .9975 -em .05 -lr .001 -C 2000 -u 3 -buffer 6000 -num 1800
"""
