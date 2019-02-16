import time
import numpy as np
import torch
import progressbar

# from unityagents import UnityEnvironment
from agent import Agent
#import utils
from utils import load_environment, load_checkpoint, print_verbose_info, report_results, get_state, print_status, save_checkpoint
from get_args import get_args

#
#
#TO-DO LIST:
#-check into implementing frame skipping for speedup and training fidelity
#-make sure that sumtree is initialized properly and not full of 0 priorities
#

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

    if not args.train:
        filepath = utils.load_filepath(args.sep) #prompt user before loading the env to avoid pop-over
        if filepath == None:
            return

    #initialize the environment
    frame_buffer = Preprocess()
    env, env_info, brain_name, nA, state_size = load_environment(args, frame_buffer)

    if args.train:
        agent = Agent(nA, state_size, args)
        print("{0}\nPrinting training data every {1} episodes.\n{0}".format(args.sep, args.print_every))
    else:
        agent = load_checkpoint(filepath, args)

    print_verbose_info(agent, env_info, args) #print extra info if flagged

    scores = run_agent(env, agent, brain_name, args, frame_buffer) #run the agent

    env.close() #close the environment

    report_results(scores, start_time) #report results

    return



def run_agent(env, agent, brain_name, args, frame_buffer):
    """Trains selected agent in the environment."""
    scores = []
    with progressbar.ProgressBar(max_value=args.print_every) as progress_bar:
        for i_episode in range(1, args.num_episodes+1):
            # reset the scenario
            score = 0
            env_info = env.reset(train_mode=args.train)[brain_name]

            # get the initial environment state
            state = get_state(env_info, agent, done=True)
            while True:
                #choose an action using current π
                action = agent.act(state)
                #use action in environment and observe results
                env_info = env.step(action.item())[brain_name]
                #collect info about new state
                reward = env_info.rewards[0]
                done = env_info.local_done[0]
                next_state = get_state(env_info, agent, done)


                # print("STATE: {}, NEXT_STATE: {}".format(state.shape, next_state.shape))

                #initiate next timestep
                agent.step(state, action, torch.tensor([reward], device=args.device), next_state, done)

                state = next_state
                score += reward
                if done:
                    break

            #prepare for next episode
            agent.update_epsilon()
            scores.append(score)
            print_status(i_episode, scores, agent, args)
            progress_bar.update(i_episode % args.print_every+1)

    save_checkpoint(agent, scores, args) #save a checkpoint if train=True

    return scores



if __name__ == "__main__":
    main()
