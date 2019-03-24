# -*- coding: utf-8 -*-
from collections import deque
import random
import torch
import numpy as np

class ReplayBuffer:
    """
    Standard replay buffer to hold memories for later learning. Returns
    random experiences with no consideration of their usefulness.

    When using an agent with a ROLLOUT trajectory, then instead of each
    experience holding SARS' data, it holds:
    state = state at t
    action = action at t
    reward = cumulative reward from t through t+n-1
    next_state = state at t+n

    where n=ROLLOUT.
    """
    def __init__(self, device, buffer_size=100000, gamma=0.99, rollout=5, agent_count=1):
        self.buffer = deque(maxlen=buffer_size)
        self.device = device
        self.gamma = gamma
        self.rollout = rollout
        self.agent_count = agent_count

    def sample(self, batch_size):
        """
        Return a sample of size BATCH_SIZE as a tuple.
        """
        batch = random.sample(self.buffer, k=batch_size)
        obs, next_obs, actions, rewards, dones = zip(*batch)

        # print("o: {}\na: {}\nr: {}\nno: {}\nd: {}".format(observations.shape, actions.shape, rewards.shape, next_observations.shape, dones.shape))

        # For actions, stack/transpose to num_agents, batch_size, action_size,
        # then combine the num_agents and action_size to get the "centralized
        # action-value function" input, outputting [centralized, batch_size]
        # for sampled experience.
        # actions = torch.stack(actions).float()
        # ashape = actions.shape
        # actions = actions.view(ashape[0], -1).to(self.device)


        # Transpose the num_agents and batch_size, for easy indexing later
        # e.g. from 64 experiences of 2 agents each, to 2 agents with 64
        # experiences each
        obs = torch.stack(observations).transpose(1,0).to(self.device)
        next_obs = torch.stack(next_observations).transpose(1,0).to(self.device)
        actions = torch.stack(actions).to(self.device)
        rewards = torch.stack(rewards).transpose(1,0).to(self.device)
        dones = torch.stack(dones).transpose(1,0).to(self.device)
        batch = (obs, next_obs, actions, rewards, dones)
        return batch

    def init_n_step(self):
        """
        Creates (or recreates to zero an existing) deque to handle nstep returns.
        """
        self.n_step = deque(maxlen=self.rollout)

    def store(self, experience):
        """
        Once the n_step memory holds ROLLOUT number of sars' tuples, then a full
        memory can be added to the ReplayBuffer.
        """
        actions = np.concatenate(actions)
        experience = obs, next_obs, actions, rewards, dones

        if self.rollout > 1:
            self.n_step.append(experience)
            # Abort if ROLLOUT steps haven't been taken in a new episode
            if len(self.n_step) < self.rollout:
                return
            experience = self._n_stack(experience)
        self.buffer.append(experience)


    def _n_stack(self, experience):
        # Unpacks and stores the SARS' tuples across ROLLOUT timesteps
        obs, next_obs, actions, rewards, dones = zip(*self.n_step)

        n_steps = self.rollout - 1
        # Calculate n-step discounted reward
        # If encountering a terminal state (next_observation == None) then sum
        # the rewards only until the terminal state and report back a terminal
        # state for the trajectory tuple.
        summed_rewards = np.zeros(self.agent_count)
        for i in range(n_steps):
            summed_rewards += self.gamma**i * rewards[i]
            if np.any(dones[i]):
                n_steps = i
                break
        # store the current state, current action, cumulative discounted
        # reward from t -> t+n-1, and the next_state at t+n (S't+n)
        #current obs [num_agents, state_size]
        obs = observations[0]
        # next_obs for use with target predictions... should be ROLLOUT steps
        # ahead of obs, unless there is a terminal state
        next_obs = next_observations[n_steps]

        #current actions [num_agents * action_size,]
        actions = torch.from_numpy(actions[0]).float()        # actions = torch.from_numpy(actions[0]).type(torch.float)

        #rewards summed through rollout-1 [num_agents,]
        summed_rewards = torch.tensor(summed_rewards).float()
        # done state  if encountered during ROLLOUT step stacking [num_agents,]
        dones = torch.tensor([dones[n_steps]])

        experience = (obs, next_obs, actions, summed_rewards, dones)
        return experience

    def __len__(self):
        return len(self.buffer)
