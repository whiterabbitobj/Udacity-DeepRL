import numpy as np
# import random
from collections import namedtuple, deque
#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim

from buffers import ReplayBuffer, nStepBuffer


class D4PG_Agent: #(Base_Agent):
    def __init__(self, state_size, action_size, agent_count, rollout=5):
        #super().__init__(self)

        self.state_size = state_size
        self.action_size = action_size
        self.agent_count = agent_count
        self.rollout = rollout

        self.memory = ReplayBuffer()

        self.actor = models.ActorNet(state_size, action_size)
        # self.critic = models.critic_net
        # self.actor_target = models.copy_actor
        # self.critic_target = models.copy_critic

        return

    def _make_actor(self):
        # Create actor network THETA
        # -make a copy theta'
        return

    def _make_critic(self):
        # Create critic network OMEGA
        # -make a copy omega'
        return

    def _launch_actors(self):
        # Copy the actor network K times and launch via multiprocessing
        return

    def act(self, states):
        actions = self.actor(states).detach()
        return


    def save_experience(self):
        pass

    def initialize_memory(self, batchsize, env):
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
        while len(self.memory) < batchsize:
            # Actions are bound [-1, 1]
            actions = np.random.rand(self.agent_count, self.action_size) * 2 - 1
            rewards, done = env.step(actions)
            next_states = env.states
            agents = [n_step_memory.experience(*i) for i in zip(states, actions, rewards, next_states)]
            n_step_memory.store(agents)

            # Once ROLLOUT consecutive actions/observations have been collected,
            # We can store experiences in the main replay buffer for learning
            if len(n_step_memory) == n_step_memory.buffer.maxlen:
                for agent in zip(*n_step_memory.buffer):
                    self.memory.store(*zip(*agent))
            states = next_states
        return
