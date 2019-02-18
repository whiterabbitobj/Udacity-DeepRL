import time
import numpy as np
import torch
import progressbar

from agent import Agent
from utils import Environment, load_checkpoint, print_verbose_info, report_results, print_status, save_checkpoint

from get_args import get_args

#
#
#TO-DO LIST:
#-check into implementing frame skipping for speedup and training fidelity, pull frames across 8-16 frames to stack four, instead of consecutive four frames
#-make sure that sumtree is initialized properly and not full of 0 priorities
#-review p_min "max weight" in PER buffer

def main():
    """
    Primary code for training or testing an agent using one of several
    optional network types. Deep Q-Network, Double DQN, etcself.

    This is a project designed for Udacity's Deep Reinforcement Learning Nanodegree
    and uses a special version of Unity's Banana learning environment. This
    environment should be available via the github repository at:
    https://github.com/whiterabbitobj/udacity-deep-reinforcement-learning/tree/master/p1_navigation
    """

    """
    Sets up a few global variables to condense code:
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
    #args.update_every = args.update_every * args.frameskip

    if not args.train:
        filepath = utils.load_filepath(args.sep) #prompt user before loading the env to avoid pop-over
        if filepath == None:
            return
    #wrap Banana environment in a Class for concise handling
    env = Environment(args)

    if args.train:
        agent = Agent(env.nA, env.state_size, args)
        print("{0}\nPrinting training data every {1} episodes.\n{0}".format(args.sep, args.print_every))
    else:
        agent = load_checkpoint(filepath, args)

    print_verbose_info(agent, args) #print extra info if flagged

    #run the agent
    scores = run_agent(agent, env, args)

    env.close() #close the environment

    report_results(scores, start_time) #report results

    return



def run_agent(agent, env, args):
    """Trains selected agent in the environment."""
    scores = []
    with progressbar.ProgressBar(max_value=args.print_every) as progress_bar:
        for i_episode in range(1, args.num_episodes+1):
            score = 0
            # reset the scenario
            env.reset()

            # get the initial environment state
            state = env.state(reset=True)

            while True:
                counter = 0
                if counter % args.frameskip == 0:
                    #choose an action using current π
                    action = agent.act(state)
                #use action in environment and observe results
                # next_state, reward, done = env.step(action.item())
                reward, done = env.step(action.item())

                #initiate next timestep
                if counter % args.frameskip == 0:
                    if done:
                        next_state = None
                    else:
                        next_state = self.state()
                    agent.step(state, action, reward, next_state)
                    state = next_state

                counter += 1
                score += reward
                if done:
                    break

            #prepare for next episode
            agent.update_epsilon(args.epsilon_decay, args.epsilon_min)
            scores.append(score)
            print_status(i_episode, scores, agent, args)
            progress_bar.update(i_episode % args.print_every+1)

    save_checkpoint(agent, scores, args) #save a checkpoint if train=True

    return scores



if __name__ == "__main__":
    main()
