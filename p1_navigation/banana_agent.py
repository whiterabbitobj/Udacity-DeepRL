import numpy as np
import torch

from unityagents import UnityEnvironment

from get_args import get_args
from agent import DQN_Agent


def main():
    args = get_args()

    #REPLACE BELOW AFTER BUG TESTING!!!
    args.mode = "train"
    args.gpu = True

    epsilon = args.epsilon
    epsilon_decay = args.epsilon_decay
    epsilon_min = args.epsilon_min

    for arg in vars(args):
        print("{}: {}".format(arg, getattr(args, arg)))
    ###################################


    #if the user has not chosen a mode, quit early and print an error
    # if not args.train and not args.run:
    #     print("ERROR: Please choose a mode in which to run the agent.")
    #     return

    #send all the training to the GPU, if available
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")

    #initialize the environment
    unity_env = UnityEnvironment(file_name="Banana_Windows_x86_64/Banana.exe")
    # get the default brain (In this environment there is only one agent/brain)
    brain_name = unity_env.brain_names[0]
    brain = unity_env.brains[brain_name]
    env = unity_env.reset(train_mode=True)[brain_name]

    action_size = brain.vector_action_space_size
    state_size = len(env.vector_observations[0])


    #print some info about the agent about to be initialized
    print('Number of agents:', len(env.agents))
    print("Observing state size: {} with {} available actions.".format(state_size, action_size))

    # THIS IS WHERE WE NEED TO IMPLEMENT DIFFERENT AGENT TYPES, THE CODE TO
    # *RUN* THE AGENT IS UNIFORM ACROSS AGENT TYPES!
    \agent = DQN_Agent(state_size, action_size, device, args, seed=0)

    num_episodes = args.episode_count
    print("Number of Episodes: {}".format(num_episodes))

    #train the agent after initializing all of the info above
    if args.mode == "train":

        avg_len = 2
        scores = []
        for i_episode in range(1, num_episodes+1):
            score = 0
            #reset the environment for a new episode runthrough
            env = unity_env.reset(train_mode=True)[brain_name]
            # get the initial environment state
            state = env.vector_observations[0]
            while True:
                #choose an action based on agent QTable
                ###action = agent.act(state, epsilon)
                action = np.random.randint(action_size)
                #use action to get updated environment state
                env = unity_env.step(action)[brain_name]
                #collect info about new state
                next_state = env.vector_observations[0]
                reward = env.rewards[0]
                done = env.local_done[0]
                #update the agent with new environment info
                ###agent.step(state, action, reward, next_state, done)
                #update current state value to choose action in next time step
                state = next_state
                #add reward from current timestep to cumulative score
                score += reward
                if done:
                    break
            #after episode completes, append the score to the list of all scores
            #in the current training session
            scores.append(score)
            #update value for  EPSILON
            epsilon = max(epsilon*epsilon_decay, epsilon_min)

            #print status info every so often
            if i_episode % avg_len == 0:
                print("Episode {}/{}, avg score for last {} episodes: {}".format(i_episode, num_episodes, avg_len, np.mean(scores[-avg_len:])))
    unity_env.close()
    return

if __name__ == "__main__":
    main()







# # examine the state space
# state = env_info.vector_observations[0]
# print('States look like:', state)
# state_size = len(state)
# print('States have length:', state_size)
#
# env_info = env.reset(train_mode=False)[brain_name] # reset the environment
# state = env_info.vector_observations[0]            # get the current state
# score = 0                                          # initialize the score
# while True:
#     action = np.random.randint(action_size)        # select an action
#     env_info = env.step(action)[brain_name]        # send the action to the environment
#     next_state = env_info.vector_observations[0]   # get the next state
#     reward = env_info.rewards[0]                   # get the reward
#     done = env_info.local_done[0]                  # see if episode has finished
#     score += reward                                # update the score
#     state = next_state                             # roll over the state to next time step
#     if done:                                       # exit loop if episode finished
#         break
#
# print("Score: {}".format(score))
#
