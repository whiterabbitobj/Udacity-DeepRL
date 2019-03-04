import numpy as np
import copy
# import random
from collections import namedtuple, deque
from models import ActorNet, CriticNet
#
import torch
# import torch.nn as nn
# import torch.nn.functional as F
import torch.optim as optim

from buffers import ReplayBuffer, nStepBuffer


class D4PG_Agent: #(Base_Agent):
    def __init__(self,
                 state_size,
                 action_size,
                 agent_count,
                 rollout=5,
                 a_lr = 1e-4,
                 c_lr = 1e-3,
                 gamma = 0.99,
                 C = 1000):
        self.t_step = 0
        self.state_size = state_size
        self.action_size = action_size
        self.agent_count = agent_count
        self.rollout = rollout
        self.gamma = gamma
        self.C = C

        self.memory = ReplayBuffer()

        self.actor = ActorNet(state_size, action_size)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=a_lr)

        self.critic = CriticNet(state_size, action_size)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=c_lr)
        self.critic_loss = nn.CrossEntropyLoss()


    def act(self, states):
        #actions = self.actor(states).detach()
        #states = torch.from_numpy(states).float()
        actions = self.actor(states).detach().numpy()
        noise = self._gauss_noise(actions.shape)
        #print("noise:", noise)
        actions += noise
        actions = np.clip(actions, -1, 1)
        return actions


    def store(self):
        pass


    def step(self, states, actions, rewards, next_states):
        agent.store(states, actions, rewards, next_states)
        agent.learn()
        self.t_step += 1


    def learn(self):
        batch = self.memory.sample()
        states, actions, rewards, next_states = batch

        target = self._get_targets(batch)
        current = self.critic(states, actions)

        #critic_loss = F.MSELoss(target, current)
        critic_loss = self.critic_loss(target, current)
        actor_loss = (self.actor(states) * current).mean()

        self.actor_optim.zero_grad()
        self.critic_optim.zero_grad()

        actor_loss.backward()
        critic_loss.backward()

        self.actor_optim.step()
        self.critic_optim.step()

        ### Either soft-update like in original DDPG
        #self._soft_update(self.critic_target, self.critic, self.tau)
        #self._soft_update(self.actor_target, self.actor, self.tau)

        ### Or hard update as in DQN and implied by D4PG paper
        if  self.t_step % self.C == 0:
            self.critic_target.load_state_dict(self.critic.state_dict())
            self.actor_target.load_state_dict(self.critic.state_dict())


    def _soft_update(self, target, active):
        for t_param, param in zip(target.parameters(), active.parameters()):
            t_param.data.copy_(tau*param.data + (1-tau)*t_param.data)

    def _get_target(self, batch):
        states, actions, rewards, next_states = batch
        for i in range(1, self.rollout+1):
            actions_calc = self.actor_target(next_states).detach()
            rewards += gamma**i *  self.critic_target(next_states, actions_calc)
        return rewards


    def _gauss_noise(self, shape, e=0.3):
        n = np.random.normal(0, 1, shape)
        return e*n

    def initialize_memory(self, pretrain_length, env):
        """
        Fills ReplayBuffer to at least the size of BATCHSIZE so that training
        can begin. Stacks each experience with ROLLOUT number of frames.

        This method uses some finnicky zips/unzips to handle data. It takes in
        SARS' data as N-row arrays (N=20 for Udacity Reacher environment), it
        changes this into N-row array of 4 columns (SARS') and stacks it in a
        deque of height=ROLLOUT (default 5) to collect rollout trajectories.

        Once the rollout length has been reached, it unpacks this deque into an
        experience trajectory that can be stored and processed in the main
        ReplayBuffer. Since there are 20 actors, 20 trajectories are stored at
        each timestep. Therefore, 20x [4, 5, [33, 4, 1, 33]] are stored, where
        each trajectory contains:
        STATES [5, 33]
        ACTIONS [5, 4]
        REWARDS [5, 1]
        NEXT_STATES [5, 33]
        """
        n_step_memory = nStepBuffer(size=self.rollout)

        # Take random actions and observe environment until replay memory has
        # enough stored experiences to fill a batch for processing
        states = env.states
        while len(self.memory) < pretain_length:
            # Actions are bound [-1, 1]
            actions = np.random.uniform(-1, 1, (self.agent_count, self.action_size))
            rewards, next_states, done = env.step(actions)
            agents = [n_step_memory.experience(*i) for i in zip(states, actions, rewards, next_states)]
            n_step_memory.store(agents)

            # Once ROLLOUT consecutive actions/observations have been collected,
            # We can store experiences in the main replay buffer for learning
            if len(n_step_memory) == n_step_memory.buffer.maxlen:
                for agent in zip(*n_step_memory.buffer):
                    self.memory.store(*zip(*agent))
            states = next_states
        return
