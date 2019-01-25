import time
import numpy as np
import torch

from unityagents import UnityEnvironment

from get_args import get_args
from agent import DQN_Agent


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

    args = get_args()

    #if the user has not chosen a mode, quit early and print an error
    if args.mode is None:
        print("ERROR: Please choose a mode in which to run the agent.")
        return

    #send all the training to the GPU, if available
    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu")

    #initialize the environment
    unity_env = UnityEnvironment(file_name="Banana_Windows_x86_64/Banana.exe", no_graphics=args.nographics)

    # get the default brain (In this environment there is only one agent/brain)
    brain_name = unity_env.brain_names[0]
    brain = unity_env.brains[brain_name]
    env = unity_env.reset(train_mode=True)[brain_name]

    action_size = brain.vector_action_space_size
    state_size = len(env.vector_observations[0])

    num_episodes = args.episode_count

    #choose how often to average the score & print training data. This value is
    #bounded between 2 and 100.
    avg_len = min(max(int(num_episodes/args.prints),2), 100)

    #PRINT DEBUG INFO
    for arg in vars(args):
        print("{}: {}".format(arg, getattr(args, arg)))
    print("#"*50)
    print("Device: {}".format(device))
    print("Action Size: {}\nState Size: {}".format(action_size, state_size))
    print('Number of agents:', len(env.agents))
    print("Number of Episodes: {}".format(num_episodes))

    # THIS IS WHERE WE NEED TO IMPLEMENT DIFFERENT AGENT TYPES, THE CODE TO
    # *RUN* THE AGENT IS UNIFORM ACROSS AGENT TYPES!
    agent = DQN_Agent(state_size, action_size, device, args, seed=0)

    #Get start time
    start_time = time.time()

    #train the agent
    if args.mode == "train":
        print("Printing training data every {} episodes.\n{}".format(avg_len,"#"*50))

        scores = []
        epsilon = args.epsilon

        for i_episode in range(1, num_episodes+1):
            score = 0
            #reset the environment for a new episode runthrough
            env = unity_env.reset(train_mode=True)[brain_name]
            # get the initial environment state
            state = env.vector_observations[0]
            while True:
                #choose an action based on agent QTable
                action = agent.act(state, epsilon)
                #action = np.random.randint(action_size)
                #use action to get updated environment state
                env = unity_env.step(action)[brain_name]
                #collect info about new state
                next_state = env.vector_observations[0]
                reward = env.rewards[0]
                done = env.local_done[0]
                #update the agent with new environment info
                agent.step(state, action, reward, next_state, done)
                #update current state value to choose action in next time step
                state = next_state
                #add reward from current timestep to cumulative score
                score += reward
                if done:
                    break
            #append current score to the scores list
            scores.append(score)
            #update value for  EPSILON
            epsilon = max(epsilon*args.epsilon_decay, args.epsilon_min)

            #print status info every so often
            if i_episode % avg_len == 0:
                print("Episode {}/{}, avg score for last {} episodes: {}".format(
                        i_episode, num_episodes, avg_len, np.mean(scores[-avg_len:])))


    print("TOTAL RUNTIME: {:.2f} seconds".format(time.time()-start_time))
    unity_env.close()
    return

if __name__ == "__main__":
    main()
