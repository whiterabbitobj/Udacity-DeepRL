import time
import numpy as np
import torch
import progressbar

from unityagents import UnityEnvironment
from get_args import get_args
from agent import Agent
import utils



def main():
    """
    Primary code for training or testing an agent using one of several
    optional network types. Deep Q-Network, Double DQN, etcself.

    This is a project designed for Udacity's Deep Reinforcement Learning Nanodegree
    and uses a special version of Unity's Banana learning environment. This
    environment should be available via the github repository at:
    https://github.com/whiterabbitobj/udacity-deep-reinforcement-learning/tree/master/p1_navigation
    """

    start_time = time.time()
    sep = "#"*50

    args = get_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if not args.train:
        filepath = utils.load_filepath(args.latest) #prompt user before loading the env to avoid pop-over
        if filepath == None:
            print("Quit before loading a file.")
            return

    #initialize the environment
    if args.pixels:
        unity_filename = "VisualBanana_Windows_x86_64/Banana.exe"
    else:
        unity_filename = "Banana_Windows_x86_64/Banana.exe"

    unity_env = UnityEnvironment(file_name=unity_filename, no_graphics=args.nographics)
    brain_name = unity_env.brain_names[0]
    brain = unity_env.brains[brain_name]
    env = unity_env.reset(train_mode=True)[brain_name]
    nA = brain.vector_action_space_size
    nS = len(env.vector_observations[0])

    #calculate how often to print status updates, min 2, max 100.
    args.print_count = utils.print_interval(args, 2, 100)


    if args.train:
        print("Printing training data every {} episodes.\n{}".format(args.print_count,sep))
        agent = Agent(nS, nA, device, args)
    else:
        agent = utils.load_checkpoint(filepath, device, args)
        args.num_episodes = 3
        agent.epsilon = 0

    if args.verbose:
        utils.print_debug_info(sep, device, nA, nS, env, args) #print info about params
        print("{}\n{}".format(agent.q, sep)) #print info about the active network


    #Run the agent
    scores = run_agent(unity_env, agent, args, brain_name)
    unity_env.close()

    print("TOTAL RUNTIME: {}.".format(utils.get_runtime(start_time)))
    utils.plot_scores(scores)

    return



def run_agent(unity_env, agent, args, brain_name):
    """Trains selected agent in the environment."""
    scores = []
    with progressbar.ProgressBar(max_value=args.print_count) as progress_bar:
        for i_episode in range(1, args.num_episodes+1):
            score = 0
            #reset the environment for a new episode runthrough
            #env = unity_env.reset(train_mode=args.train)[brain_name]
            env = unity_env.reset(train_mode=True)[brain_name]

            # get the initial environment state
            state = env.visual_observations[0].squeeze(0).transpose(2,0,1) if args.pixels else env.vector_observations[0]

            while True:
                #choose an action use current policy and take a timestep using this action
                action = agent.act(state)
                env = unity_env.step(action)[brain_name]

                #collect info about new state
                reward = env.rewards[0]
                next_state = env.visual_observations[0].squeeze(0).transpose(2,0,1) if args.pixels else env.vector_observations[0]
                done = env.local_done[0]
                score += reward

                #initiate next timestep
                if args.train:
                    agent.step(state, action, reward, next_state, done)

                state = next_state
                if done:
                    break
            agent.update_epsilon() #epsilon is 0 in evaluation mode
            #prepare for next episode
            scores.append(score)
            utils.print_status(i_episode, scores, args, agent)
            progress_bar.update(i_episode%args.print_count+1)

    if args.train:
        print(agent.t_step)
        utils.save_checkpoint(agent, scores, args.print_count)
    return scores



if __name__ == "__main__":
    main()
