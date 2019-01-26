import time
import random
import os.path
from collections import namedtuple, deque

import numpy as np
import torch



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

    for i_episode in range(1, args.num_episodes+1):
        score = 0
        #reset the environment for a new episode runthrough
        env = unity_env.reset(train_mode=True)[brain_name]
        # get the initial environment state
        state = env.vector_observations[0]
        while True:
            #choose an action use current policy and take a timestep using this action
            #action = agent.act(state, epsilon)
            action = agent.act(state)

            env = unity_env.step(action)[brain_name]

            #collect info about new state
            next_state = env.vector_observations[0]
            reward = env.rewards[0]
            done = env.local_done[0]
            score += reward

            #initiate next timestep
            agent.step(state, action, reward, next_state, done)
            state = next_state

            if done:
                break

        agent.update_epsilon()

        #prepare for next episode
        scores.append(score)
        print_status(i_episode, scores, args)
    save_name = "checkpoint_" + agent.name + time.strftime("_%Y_%m_%d_%Hh%Mm%Ss", time.gmtime()) + ".pth"
    save_checkpoint(agent,save_name)
    return scores



def save_checkpoint(agent, save_name):
    '''
    Saves the current Agent's learning dict as well as important parameters
    involved in the latest training.
    '''
    agent.qnet_local.to('cpu')
    checkpoint = {'agent_type': agent.name,
                  'state_size': agent.nS,
                  'action_size': agent.nA,
                  'state_dict': agent.qnet_local.state_dict(),
                  'optimizer': agent.optimizer.state_dict()
                  }
    # checkpoint = {'agent_type': agent.name,
    #               'drop_rate': agent.dropout,
    #               'state_dict': agent.qnet_local.state_dict(),
    #               'lr': agent.lr,
    #               'tau': agent.tau,
    #               'gamma': agent.gamma,
    #               'batchsize': agent.batchsize,
    #               'buffersize': agent.buffersize,
    #               'epsilon': agent.epsilon,
    #               'state_size': agent.nS,
    #               'action_size': agent.nA,
    #               'dropout': agent.dropout,
    #               'optimizer': agent.optimizer.state_dict()
    #               }
    torch.save(checkpoint, save_name)
    return True



def get_latest_file():
    return



def print_debug_info(device, action_size, state_size, env, args):
    print("#"*50)
    for arg in vars(args):
        print("{}: {}".format(arg, getattr(args, arg)))
    print("#"*50)
    print("Device: {}".format(device))
    print("Action Size: {}\nState Size: {}".format(action_size, state_size))
    print('Number of agents:', len(env.agents))
    print("Number of Episodes: {}".format(args.num_episodes))



def print_status(i_episode, scores, args):
    if i_episode % args.print_count == 0:
        print("Episode {}/{}, avg score for last {} episodes: {}".format(
                i_episode, args.num_episodes, args.print_count, np.mean(scores[-args.print_count:])))
