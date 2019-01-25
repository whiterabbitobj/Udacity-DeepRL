import numpy as np
import random
import torch

from collections import namedtuple, deque


class ReplayBuffer:
    def __init__(self, nA, buffersize, batchsize, seed, device):
        self.nA = nA
        self.memory = deque(maxlen=buffersize)
        self.batchsize = batchsize
        self.item = namedtuple("Item", field_names=['state','action','reward','next_state','done'])
        self.seed = random.seed(seed)
        self.device = device

    def add(self, state, action, reward, next_state, done):
        t = self.item(state, action, reward, next_state, done)
        self.memory.append(t)

    def sample(self):
        batch = random.sample(self.memory, k=self.batchsize)

        states = torch.from_numpy(np.vstack([item.state for item in batch if item is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([item.action for item in batch if item is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([item.reward for item in batch if item is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([item.next_state for item in batch if item is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([item.done for item in batch if item is not None]).astype(np.uint8)).float().to(self.device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)



def train(unity_env, agent, args, brain_name):
    """Trains selected agent in the environment."""

    scores = []
    epsilon = args.epsilon
    print("Printing training data every {} episodes.\n{}".format(args.print_count,"#"*50))

    for i_episode in range(1, args.num_episodes+1):
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
        if i_episode % args.print_count == 0:
            print("Episode {}/{}, avg score for last {} episodes: {}".format(
                    i_episode, args.num_episodes, args.print_count, np.mean(scores[-args.print_count:])))
    return scores
