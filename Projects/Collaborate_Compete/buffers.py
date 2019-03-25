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
        # Transpose the num_agents and batch_size, for easy indexing later
        # e.g. from 64 experiences of 2 agents each, to 2 agents with 64
        # experiences each
        obs = torch.stack(obs).transpose(1,0).to(self.device)
        next_obs = torch.stack(next_obs).transpose(1,0).to(self.device)
        actions = torch.stack(actions).to(self.device)
        rewards = torch.stack(rewards).transpose(1,0).to(self.device)
        dones = torch.stack(dones).transpose(1,0).to(self.device)
        return (obs, next_obs, actions, rewards, dones)

    def init_n_step(self):
        """
        Creates (or rezeroes an existing) deque to handle nstep returns.
        """
        self.n_step = deque(maxlen=self.rollout)

    def store(self, obs, next_obs, actions, rewards, dones):
        """
        Once the n_step memory holds ROLLOUT number of sars' tuples, then a full
        memory can be added to the ReplayBuffer.
        """
        actions = np.concatenate(actions)

        if self.rollout > 1:
            experience = obs, next_obs, actions, rewards, dones
            self.n_step.append(experience)
            # Abort if ROLLOUT steps haven't been taken in a new episode
            if len(self.n_step) < self.rollout:
                return
            obs, next_obs, actions, rewards, dones = self._n_stack()

        actions = torch.from_numpy(actions).float()
        rewards = torch.tensor(rewards).float()
        dones = torch.tensor(dones)

        experience = (obs, next_obs, actions, rewards, dones)
        self.buffer.append(experience)


    def _n_stack(self):
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
        obs = obs[0]
        # next_obs is ROLLOUT steps ahead of obs, unless there is terminal state
        next_obs = next_obs[n_steps]
        actions = actions[0]
        dones = dones[n_steps]
        return obs, next_obs, actions, summed_rewards, dones

    def __len__(self):
        return len(self.buffer)
