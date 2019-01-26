import time
import numpy as np
import torch

from unityagents import UnityEnvironment

from get_args import get_args
from agent import DQN_Agent
from agent_utils import train
from utils import load_filepath, load_checkpoint, plot_scores, print_debug_info, print_status



def main():
    """
    Primary code for training or testing an agent using one of several
    optional network types. Deep Q-Network, Double DQN, etcself.

    This is a project designed for Udacity's Deep Reinforcement Learning Nanodegree
    and uses a special version of Unity's Banana learning environment. This
    environment should be available via the github repository at:
    https://github.com/whiterabbitobj/udacity-deep-reinforcement-learning/tree/master/p1_navigation

    Example command:
    python banana_deeprl.py -mode train -a DDQN --epsilon .9 --epsilon_decay .978
    """

    #Get start time
    start_time = time.time()
    #gather parameters
    args = get_args()

    #prep any params that need calculation

    #calculate how often to print status updates, min 2, max 100.
    args.print_count = min(max(int(args.num_episodes/args.print_count),2), 100)

    #send all the training to the GPU, if available
    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu")

    #prep data based on mode
    if args.mode == "demo":
        filepath = load_filepath(args.latest)
        agent = load_checkpoint(filepath, device, args)
    elif args.mode == "train":
        print("Printing training data every {} episodes.\n{}".format(args.print_count,"#"*50))
    else:
        print("ERROR: Please choose a mode in which to run the agent.")
        return


    #initialize the environment
    unity_env = UnityEnvironment(file_name="Banana_Windows_x86_64/Banana.exe", no_graphics=args.nographics)
    # get the default brain (In this environment there is only one agent/brain)
    brain_name = unity_env.brain_names[0]
    brain = unity_env.brains[brain_name]
    env = unity_env.reset(train_mode=True)[brain_name]
    nA = brain.vector_action_space_size
    nS = len(env.vector_observations[0])
    if args.debug:
        print_debug_info(device, nA, nS, env, args)


    #Run the agent
    if args.mode == "demo":
        env = unity_env.reset(train_mode=False)[brain_name]
    elif args.mode == "train":
        # THIS IS WHERE WE NEED TO IMPLEMENT DIFFERENT AGENT TYPES, THE CODE TO
        # *RUN* THE AGENT IS UNIFORM ACROSS AGENT TYPES!
        agent = DQN_Agent(nS, nA, device, args)
        scores = train(unity_env, agent, args, brain_name)
        plot_scores(scores)
    else:
        print("Something went wrong trying to execute the agent. Neither TRAIN/DEMO executed correctly.")


    print("TOTAL RUNTIME: {:.1f} seconds".format(time.time()-start_time))
    unity_env.close()
    return


if __name__ == "__main__":
    main()
