import time
import numpy as np
import torch
import progressbar

# from unityagents import UnityEnvironment
from agent import Agent
import utils
from get_args import get_args



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
    args = get_args()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.print_every = np.clip(args.num_episodes//args.print_count, 2, 100)
    args.sep = "#"*50
    args.start_time = time.time()

    if not args.train:
        filepath = utils.load_filepath(args.sep) #prompt user before loading the env to avoid pop-over
        if filepath == None:
            return

    #initialize the environment
    env, env_info, brain_name, nA, nS = utils.load_environment(args)
    print(env)
    if args.train:
        agent = Agent(nS, nA, args)
        print("Printing training data every {} episodes.\n{}".format(args.print_every, args.sep))
    else:
        agent = utils.load_checkpoint(filepath, args)

    utils.print_verbose_info(agent, env, env_info, args)

    scores = run_agent(env, agent, brain_name, args) #Run the agent

    env.close() #close the environment

    utils.report_results(scores) #report results

    return



def run_agent(env, agent, brain_name, args):
    """Trains selected agent in the environment."""
    scores = []
    with progressbar.ProgressBar(max_value=args.print_count) as progress_bar:
        for i_episode in range(1, args.num_episodes+1):
            score = 0

            env_info = env.reset(train_mode=args.train)[brain_name]

            # get the initial environment state
            if args.pixels:
                state = env_info.visual_observations[0].squeeze(0).transpose(2,0,1)
            else:
                state = env_info.vector_observations[0]

            while True:
                #choose an action using current π
                action = agent.act(state)
                env_info = env.step(action)[brain_name]
                #collect info about new state
                reward = env_info.rewards[0]
                next_state = env_info.visual_observations[0].squeeze(0).transpose(2,0,1) if args.pixels else env_info.vector_observations[0]
                done = env_info.local_done[0]
                score += reward
                #initiate next timestep
                if args.train:
                    agent.step(state, action, reward, next_state, done)

                state = next_state
                if done:
                    break
            #prepare for next episode
            scores.append(score)
            utils.print_status(i_episode, scores, agent)
            progress_bar.update(i_episode % args.print_count+1)

    if args.train:
        print(agent.t_step)
        utils.save_checkpoint(agent, scores)

    return scores



if __name__ == "__main__":
    main()
