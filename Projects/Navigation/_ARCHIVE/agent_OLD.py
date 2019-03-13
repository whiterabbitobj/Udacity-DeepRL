import numpy as np
import random
from collections import namedtuple, deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Agent():
    """Uses a classic Deep Q-Network to learn from the environment"""

    def __init__(self, nA, state_size, args, seed=0):
        #super(DQN_Agent, self).__init()

        #initialize agent parameters
        #self.nS = state_size
        self.nA = nA
        self.seed = 0#random.seed(seed)
        self.framestack = args.framestack
        self.device = args.device
        self.t_step = 0
        #initialize params from the command line args
        self.framework = args.framework
        self.batchsize = args.batchsize
        #self.buffersize = args.buffersize
        self.C = args.C
        #self.dropout = args.dropout
        self.epsilon = args.epsilon
        self.gamma = args.gamma
        #self.lr = args.learn_rate
        self.update_every = args.update_every
        #self.momentum = args.momentum
        #self.no_per = args.no_prioritized_replay
        self.train = args.train
        self.pixels = args.pixels

        #Initialize Q-Network
        self.q = self._make_model(args.pixels, state_size, args.dropout)
        self.qhat = self._make_model(args.pixels, state_size, args.dropout)
        self.qhat.load_state_dict(self.q.state_dict())
        self.qhat.eval()
        self.optimizer = self._set_optimizer(self.q.parameters(), args.optimizer, args.learn_rate, args.momentum)

        #initialize REPLAY buffer
        if args.no_prioritized_replay:
            self.memory = ReplayBuffer(args.buffersize, self.framestack, self.device, state_size, self.pixels)
        else:
            self.memory = PERBuffer(args.buffersize, self.batchsize, self.framestack, self.device, args.alpha, args.beta)
            self.criterion = WeightedLoss()

    # def act(self, state):
    #     """Select an action using epsilon-greedy π.
    #        Always use greedy if not training.
    #     """
    #     if random.random() > self.epsilon or not self.train:
    #         self.q.eval()
    #         with torch.no_grad():
    #             action_values = self.q(state)#.detach()
    #         self.q.train()
    #         action = action_values.max(1)[1].view(1,1)
    #         #print("Using greedy action:", action.item())
    #         return action
    #     else:
    #         #print("Using random action.")
    #         return torch.tensor([[random.randrange(self.nA)]], device=self.device, dtype=torch.long)

    def step(self, state, action, reward, next_state):
        """Moves the agent to the next timestep.
           Learns each UPDATE_EVERY steps.
        """
        if not self.train:
            return
        reward = torch.tensor([reward], device=self.device)
        self.memory.store(state, action, reward, next_state)

        if self._memory_loaded(): # and self.t_step % self.update_every == 0:
            self.learn()
        # ------------------- update target network ------------------- #
        # self._soft_update(self.q, self.qhat, 0.001)
        #update the target network every C steps
        if self.t_step % self.C == 0:
            self.qhat.load_state_dict(self.q.state_dict())

        self.t_step += 1



    def _soft_update(self, local_model, target_model, tau):
        """
        Soft update model parameters.
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)



    def learn(self):
        """
        Trains the Deep QNetwork and returns action values.
        Can use multiple frameworks.
        """
        #If using standard ReplayBuffer, is_weights & tree_idx will return None
        batch, is_weights, tree_idx = self.memory.sample(self.batchsize)

        state_batch = torch.cat(batch.state) #[64,1]
        action_batch = torch.cat(batch.action) #[64,1]
        reward_batch = torch.cat(batch.reward) #[64,1]
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device, dtype=torch.uint8)
        next_states = torch.cat([s for s in batch.next_state if s is not None])

        qhat_next_values = torch.zeros(self.batchsize, device=self.device) #[64]
        if self.framework == 'DQN':
            #VANILLA DQN: get max predicted Q values for the next states from the target model
            qhat_next_values[non_final_mask] = self.qhat(next_states).detach().max(1)[0] #[64]

        if self.framework == 'DDQN':
            #DOUBLE DQN: get maximizing action under Q, evaluate actionvalue under qHAT
            q_next_actions = self.q(next_states).detach().argmax(1)
            qhat_next_values[non_final_mask] = self.qhat(next_states).gather(1, q_next_actions.unsqueeze(1)).squeeze(1)

        expected_values = reward_batch + (self.gamma * qhat_next_values) #[64]

        expected_values = expected_values.unsqueeze(1) #[64,1]
        values = self.q(state_batch).gather(1, action_batch) #[64,1]

        if is_weights is None:
            #Huber Loss provides better results than MSE
            loss = F.smooth_l1_loss(values, expected_values) #[64,1]
        else:
            #Compute Huber Loss manually to utilize is_weights
            loss, td_errors = self.criterion.huber(values, expected_values, is_weights)
            self.memory.batch_update(tree_idx, td_errors)

        #backpropogate
        self.optimizer.zero_grad()
        loss.backward()
        # for param in self.q.parameters():
        #     param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def update_epsilon(self, ed, em):
        self.epsilon = max(em, self.epsilon * ed)

    # def _set_optimizer(self, params, optimizer, lr, momentum):
    #     """
    #     Sets the optimizer based on command line choice. Defaults to Adam.
    #     """
    #     if optimizer == "RMSprop":
    #         return optim.RMSprop(params, lr=lr, momentum=momentum)
    #     elif optimizer == "SGD":
    #         return optim.SGD(params, lr=lr, momentum=momentum)
    #     else:
    #         return optim.Adam(params, lr=lr)
    #
    # def _make_model(self, use_cnn, state_size, dropout):
    #     """
    #     Sets up the network model based on whether state data or pixel data is
    #     provided.
    #     """
    #     if use_cnn:
    #         return QCNNetwork(state_size, self.nA, self.seed).to(self.device)
    #     else:
    #         return QNetwork(state_size, self.nA, self.seed, dropout).to(self.device)

    def _memory_loaded(self):
        """Determines whether it's safe to start learning because the memory is
           sufficiently filled.
        """
        if self.memory.type == "ReplayBuffer":
            if len(self.memory) >= self.batchsize:
                return True
        if self.memory.type == "PERBuffer":
            if self.memory.tree.num_entries >= self.batchsize:
                return True
        return False
