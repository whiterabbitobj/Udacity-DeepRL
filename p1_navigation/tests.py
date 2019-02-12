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
    """
    Primary code for training or testing an agent using one of several
    optional network types. Deep Q-Network, Double DQN, etcself.

    This is a project designed for Udacity's Deep Reinforcement Learning Nanodegree
    and uses a special version of Unity's Banana learning environment. This
    environment should be available via the github repository at:
    https://github.com/whiterabbitobj/udacity-deep-reinforcement-learning/tree/master/p1_navigation
    """

    """Sets up a few global variables to condense code:
            args - arguments from command line, including defaults
            sep - separator used for print statements
            start_time - start time of program initialization
            device - which device to run on (usually GPU)
    """
    start_time = time.time()
    args = get_args()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.print_every = np.clip(args.num_episodes//args.print_count, 2, 100)
    args.sep = "#"*50

    # if not args.train:
    #     filepath = utils.load_filepath(args.sep) #prompt user before loading the env to avoid pop-over
    #     if filepath == None:
    #         return

    #initialize the environment
    env, env_info, brain_name, nA, nS = utils.load_environment(args)
    print("*"*100)
    # print("ENV:", env)
    # print("ENV_INFO:", env_info)
    # print("BRAIN NAME:",brain_name)
    # print("NA:", nA)
    # print("NS:", nS)

    state = env_info.visual_observations[0].squeeze(0).astype(np.float32).transpose(2,0,1)# / 255#.transpose(2,0,1)
    state = torch.from_numpy(state)
    #print(state.shape)
    #state = torch.from_numpy(state).float().unsqueeze(0).to(args.device)
    print(state.shape, type(state), state.dtype)
    fit = T.Compose([T.ToPILImage(), T.Grayscale(),T.ToTensor()])
    frame = fit(state).to(args.device)
    print(frame.shape, frame.cuda())
    #frame = Image.fromarray(state)

    env.close() #close the environment


if __name__ == "__main__":
    main()
