import time
import numpy as np
import torch

from unityagents import UnityEnvironment

from get_args import get_args
from agent import DQN_Agent
from agent_utils import run_agent
from utils import load_filepath, load_checkpoint, plot_scores, print_debug_info



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
    sep = "#"*50
    #gather parameters
    args = get_args()


    #send all the training to the GPU, if available
    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu")

    if not args.train:
        filepath = load_filepath(args.latest) #prompt user before loading the env to avoid pop-over
        if filepath == None:
            print("Quit before loading a file.")
            return
    #initialize the environment
    unity_env = UnityEnvironment(file_name="Banana_Windows_x86_64/Banana.exe", no_graphics=args.nographics)
    # get the default brain (In this environment there is only one agent/brain)
    brain_name = unity_env.brain_names[0]
    brain = unity_env.brains[brain_name]
    env = unity_env.reset(train_mode=True)[brain_name]
    nA = brain.vector_action_space_size
    nS = len(env.vector_observations[0])

    #calculate how often to print status updates, min 2, max 100.
    args.print_count = min(max(int(args.num_episodes/args.print_count),2), 100)


    if args.debug:
        print_debug_info(sep, device, nA, nS, env, args) #print info about params


    if args.train:
        print("Printing training data every {} episodes.\n{}".format(args.print_count,sep))
        # THIS IS WHERE WE NEED TO IMPLEMENT DIFFERENT AGENT TYPES, THE CODE TO
        # *RUN* THE AGENT IS UNIFORM ACROSS AGENT TYPES!
        agent = DQN_Agent(nS, nA, device, args)
    else:
        agent = load_checkpoint(filepath, device, args)
        args.num_episodes = 3
        agent.epsilon = 0

    if args.verbose:
        print("{}\n{}".format(agent.q, sep)) #print info about the active network


    #Run the agent
    scores = run_agent(unity_env, agent, args, brain_name)
    plot_scores(scores)

    runtime = time.time() - start_time
    m, s = divmod(runtime, 60)
    h, m = divmod(m, 60)
    #print("TOTAL RUNTIME: {:.1f} seconds".format(time.time()-start_time))
    print("TOTAL RUNTIME: {}h{}m{}s.".format(int(h), int(m), int(s)))
    unity_env.close()
    return


if __name__ == "__main__":
    main()
