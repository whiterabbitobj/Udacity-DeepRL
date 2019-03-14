# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from buffers import ReplayBuffer
from models import QNetwork



class DQN_Agent:
    """
    PyTorch Implementation of DQN/DDQN.
    """
    def __init__(self,
                 state_size,
                 action_size,
                 args,
                ):
        """
        Initialize a D4PG Agent.
        """

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.action_size = action_size
        self.state_size = state_size
        self.framework = args.framework
        self.eval = args.eval
        self.agent_count = 1
        self.learn_rate = .0001 #0.0001
        self.batch_size = 64
        self.buffer_size = 30000
        self.C = 650#*2.4
        self._epsilon = 1
        self.epsilon_decay = .9999
        self.epsilon_min = 0.01

        self.gamma = 0.99
        self.rollout = 5
        self.tau = 0.0005
        self.momentum = 1
        self.l2_decay = 0.0001
        self.update_type = "hard"


        self.t_step = 0
        self.episode = 0
        self.seed = 0

        # Set up memory buffers
        if args.prioritized_experience_replay:
            self.memory = PERBuffer(args.buffersize, self.batchsize, self.framestack, self.device, args.alpha, args.beta)
            self.criterion = WeightedLoss()
        else:
            self.memory = ReplayBuffer(self.device, self.buffer_size, self.gamma, self.rollout)

        #                    Initialize Q networks                         #
        self.q = self._make_model(state_size, action_size, args.pixels)
        self.q_target = self._make_model(state_size, action_size, args.pixels)
        self._hard_update(self.q, self.q_target)
        self.q_optimizer = self._set_optimizer(self.q.parameters(), lr=self.learn_rate, decay=self.l2_decay, momentum=self.momentum)


        self.new_episode()


    @property
    def epsilon(self):
        """
        This property ensures that the annealing process is run every time that
        E is called.
        Anneals the epsilon rate down to a specified minimum to ensure there is
        always some noisiness to the policy actions. Returns as a property.
        """

        self._epsilon = max(self.epsilon_min, self.epsilon_decay ** self.t_step)
        # self._epsilon = max(self.epsilon_min, self.epsilon_decay * self._epsilon)

        return self._epsilon

    def act(self, state, eval=False, pretrain=False):
        """
        Select an action using epsilon-greedy π.
        Always use greedy if not training.
        """

        if np.random.random() > self.epsilon or not eval and not pretrain:
            state = state.to(self.device)
            #self.q.eval()
            with torch.no_grad():
                action_values = self.q(state).detach().cpu()
            #self.q.train()
            action = action_values.argmax(dim=1).unsqueeze(0).numpy()
        else:
            action = np.random.randint(self.action_size, size=(1,1))
        return action.astype(np.long)

    def step(self, state, action, reward, next_state, pretrain=False):
        """
        Add the current SARS' tuple into the short term memory, then learn
        """

        # Current SARS' stored in short term memory, then stacked for NStep
        experience = (state, action, reward, next_state)
        # print("SENDING MEMORY:")
        # print(experience)
        if self.rollout == 1:
            self.memory.store_trajectory(state, torch.from_numpy(action), torch.tensor([reward]), next_state)
        else:
            self.memory.store_experience(experience)
        self.t_step += 1

        # Learn after done pretraining
        if not pretrain:
            self.learn()

    def learn(self):
        """
        Trains the Deep QNetwork and returns action values.
        Can use multiple frameworks.
        """

        # Sample from replay buffer, REWARDS are sum of (ROLLOUT - 1) timesteps
        # Already calculated before storing in the replay buffer.
        # NEXT_STATES are ROLLOUT steps ahead of STATES
        batch, is_weights, tree_idx = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, terminal_mask = batch

        q_values = torch.zeros(self.batch_size).to(self.device)
        if self.framework == 'DQN':
            # Max predicted Q values for the next states from the target model
            q_values[terminal_mask] = self.q_target(next_states).detach().max(dim=1)[0]

        if self.framework == 'DDQN':
            # Get maximizing ACTION under Q, evaluate actionvalue
            # under q_target

            # Max valued action under active network
            max_actions = self.q(next_states).detach().argmax(1).unsqueeze(1)
            # Use the active network action to get the value of the stable
            # target network
            q_values[terminal_mask] = self.q_target(next_states).detach().gather(1, max_actions).squeeze(1)
            # q_values[terminal_mask] = self.q_target(next_states).gather(1, max_actions).squeeze(1)

        targets = rewards + (self.gamma**self.rollout * q_values)

        targets = targets.unsqueeze(1)
        values = self.q(states).gather(1, actions)

        #Huber Loss provides better results than MSE
        if is_weights is None:
            #loss = F.smooth_l1_loss(values, targets)
            loss = F.smooth_l1_loss(values, targets)

        #Compute Huber Loss manually to utilize is_weights with Prioritization
        else:
            loss, td_errors = self.criterion.huber(values, targets, is_weights)
            self.memory.batch_update(tree_idx, td_errors)

        # Perform gradient descent
        self.q_optimizer.zero_grad()
        loss.backward()
        self.q_optimizer.step()

        self._update_networks()
        self.loss = loss.item()

    def initialize_memory(self, pretrain_length, env):
        """
        Fills up the ReplayBuffer memory with PRETRAIN_LENGTH number of experiences
        before training begins.
        """

        if len(self.memory) >= pretrain_length:
            print("Memory already filled, length: {}".format(len(self.memory)))
            return

        print("Initializing memory buffer.")

        while True:
            done = False
            env.reset()
            state = env.state
            while not done:
                action = self.act(state, pretrain=True)
                next_state, reward, done = env.step(action)
                if done:
                    next_state = None

                self.step(state, action, reward, next_state, pretrain=True)
                states = next_state

                if self.t_step % 50 == 0 or len(self.memory) >= pretrain_length:
                    print("Taking pretrain step {}... memory filled: {}/{}\
                        ".format(self.t_step, len(self.memory), pretrain_length))
                if len(self.memory) >= pretrain_length:
                    print("Done!")
                    self.t_step = 0
                    self._epsilon = 1
                    return

    def new_episode(self):
        """
        Handle any cleanup or steps to begin a new episode of training.
        """

        self.memory.init_n_step()
        self.episode += 1

    def _update_networks(self):
        """
        Updates the network using either DDPG-style soft updates (w/ param \
        TAU), or using a DQN/D4PG style hard update every C timesteps.
        """

        if self.update_type == "soft":
            self._soft_update(self.q, self.q_target)
        elif self.t_step % self.C == 0:
            self._hard_update(self.q, self.q_target)

    def _soft_update(self, active, target):
        """
        Slowly updated the network using every-step partial network copies
        modulated by parameter TAU.
        """

        for t_param, param in zip(target.parameters(), active.parameters()):
            t_param.data.copy_(self.tau*param.data + (1-self.tau)*t_param.data)

    def _hard_update(self, active, target):
        """
        Fully copy parameters from active network to target network. To be used
        in conjunction with a parameter "C" that modulated how many timesteps
        between these hard updates.
        """

        target.load_state_dict(active.state_dict())

    def _set_optimizer(self, params,lr, decay, momentum,  optimizer="Adam"):
        """
        Sets the optimizer based on command line choice. Defaults to Adam.
        """

        if optimizer == "RMSprop":
            return optim.RMSprop(params, lr=lr, momentum=momentum)
        elif optimizer == "SGD":
            return optim.SGD(params, lr=lr, momentum=momentum)
        else:
            return optim.Adam(params, lr=lr, weight_decay=decay)

    def _make_model(self, state_size, action_size, use_cnn):
        """
        Sets up the network model based on whether state data or pixel data is
        provided.
        """

        if use_cnn:
            return QCNNetwork(state_size, action_size, self.seed).to(self.device)
        else:
            return QNetwork(state_size, action_size, self.seed).to(self.device)
