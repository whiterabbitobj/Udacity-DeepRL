import time
import random
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

#     torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')


def save_checkpoint(agent, save_name):
    '''
    Saves the current Agent's learning dict as well as important parameters
    involved in the latest training.
    '''
    checkpoint = {'agent_type': agent.name,
                  'drop_rate': agent.dropout,
                  'state_dict': agent.qnet_local.state_dict(),
                  'lr': agent.lr,
                  'tau': agent.tau,
                  'gamma': agent.gamma,
                  'batchsize': agent.batchsize,
                  'buffersize': agent.buffersize,
                  'epsilon': agent.epsilon,
                  'state_size': agent.nS,
                  'action_size': agent.nA,
                  'dropout': agent.dropout,
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
    #               'input_size': agent.qnet_local.hidden_layers[0].in_features,
    #               'output_size': agent.qnet_local.output.out_features,
    #               'dropout': agent.dropout,
    #               'optimizer': agent.optimizer.state_dict()
    #               }
    agent.qnet_local.to('cpu')
    agent.qnet_target.to('cpu')
    torch.save(checkpoint, save_name)
    return True



def load_checkpoint(filepath):
    '''Loads a checkpoint from an earlier trained model.
        Requires these custom sub-attributes to be present in the save file:
        arch
        optimizer
        epochs
    '''

    checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage)

    if checkpoint['agent_type'] == 'DQN':
        agent = DQN_Agent(state_size, action_size, device, args)
    agent.qnet_local.load_state_dict(checkpoint['state_dict'])
    agent.optimizer.load_state_dict(checkpoint['optimizer'])

    if checkpoint['arch'] == 'densenet121':
        model = models.densenet121(pretrained=True)
        model.name = checkpoint['arch']
    elif checkpoint['arch'] == 'densenet169':
        model = models.densenet169(pretrained=True)
        model.name = checkpoint['arch']
    elif checkpoint['arch']  == 'vgg16':
        model = models.vgg16(pretrained=True)
    else:
        print("Sorry, this checkpoint asks for a model that isn't supported! ({})".format(checkpoint['arch']))

    model.classifier = Network(checkpoint['input_size'],
                               checkpoint['output_size'],
                               checkpoint['hidden_layers'],
                               checkpoint['drop_rate']
                               )
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    model.lr = checkpoint['lr']
    model.epochs = checkpoint['epochs']

    optimizer = optim.Adam(model.classifier.parameters(), lr=model.lr)
    optimizer.load_state_dict(checkpoint['optimizer'])

    return model



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
