import time
import numpy as np
import torch

from unityagents import UnityEnvironment

from get_args import get_args
from agent import Agent
from utils import load_filepath, load_checkpoint, plot_scores, print_debug_info, print_status, save_checkpoint
import progressbar



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
        agent = Agent(nS, nA, device, args)
    else:
        agent = load_checkpoint(filepath, device, args)
        args.num_episodes = 3
        agent.epsilon = 0

    if args.verbose:
        print("{}\n{}".format(agent.q, sep)) #print info about the active network


    #Run the agent
    scores = run_agent(unity_env, agent, args, brain_name)

    m, s = divmod(time.time() - start_time, 60)
    h, m = divmod(m, 60)
    print("TOTAL RUNTIME: {}h{}m{}s.".format(int(h), int(m), int(s)))
    plot_scores(scores)

    unity_env.close()
    return



def run_agent(unity_env, agent, args, brain_name):
    """Trains selected agent in the environment."""
    scores = []
    with progressbar.ProgressBar(max_value=args.print_count) as progress_bar:
        for i_episode in range(1, args.num_episodes+1):
            score = 0
            #reset the environment for a new episode runthrough
            env = unity_env.reset(train_mode=args.train)[brain_name]
            # get the initial environment state
            state = env.vector_observations[0]

            while True:
                #choose an action use current policy and take a timestep using this action
                action = agent.act(state)
                env = unity_env.step(action)[brain_name]

                #collect info about new state
                reward = env.rewards[0]
                next_state = env.vector_observations[0]
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
            print_status(i_episode, scores, args, agent)
            progress_bar.update(i_episode%args.print_count+1)

    if args.train:
        save_checkpoint(agent, scores, args.print_count)
    return scores



if __name__ == "__main__":
    main()
