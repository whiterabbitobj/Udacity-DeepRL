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
    def __init__(self, device, buffer_size=100000, gamma=0.99, rollout=5):
        self.buffer = deque(maxlen=buffer_size)
        self.device = device
        self.gamma = gamma
        self.rollout = rollout

    # def store_trajectory(self, observations, actions, rewards, next_observations):
    #     """
    #     Stores a trajectory, which may or may not be the same as an experience,
    #     but allows for n_step rollout.
    #     """
    #
    #     trajectory = (observations, actions, rewards, next_observations)
    #     self.buffer.append(trajectory)

    def sample(self, batch_size):
        """
        Return a sample of size BATCH_SIZE as a tuple.
        """
        batch = random.sample(self.buffer, k=batch_size)
        observations, actions, rewards, next_observations = zip(*batch)

        observations = torch.cat(observations).to(self.device)
        actions = torch.cat(actions).to(self.device).long()
        rewards = torch.cat(rewards).to(self.device)
        terminal_mask = torch.tensor(tuple(map(lambda s: s is not None, next_observations)), dtype=torch.uint8).to(self.device)
        next_observations = torch.cat([s for s in next_observations if s is not None]).to(self.device)

        batch = (states, actions, rewards, next_states, terminal_mask)
        return batch

    def init_n_step(self):
        """
        Creates (or recreates to zero an existing) deque to handle nstep returns.
        """
        self.n_step = deque(maxlen=self.rollout)

    def store_trajectory(self, experience):
        """
        Once the n_step memory holds ROLLOUT number of sars' tuples, then a full
        memory can be added to the ReplayBuffer.
        """
        if self.rollout > 1:
            self.n_step.append(experience)
            # Abort if ROLLOUT steps haven't been taken in a new episode
            if len(self.n_step) < self.rollout:
                return

            # Unpacks and stores the SARS' tuples across ROLLOUT timesteps
            observations, actions, rewards, next_observations = zip(*self.n_step)
            n_steps = self.rollout - 1

            # Calculate n-step discounted reward
            # If encountering a terminal state (next_observation == None) then sum
            # the rewards only until the terminal state and report back a terminal
            # state for the trajectory tuple.
            summed_rewards = np.zeros(len(rewards[0]))
            for i in range(n_steps):
                summed_rewards += self.gamma**i * np.array(rewards[i])
                if next_observations[i] is None:
                    n_steps = i
                    break
                # else:
                    # r += self.gamma**i * rewards[i]
            # rewards = r

            # store the current state, current action, cumulative discounted
            # reward from t -> t+n-1, and the next_state at t+n (S't+n)
            observations = observations[0]
            actions = torch.from_numpy(actions[0])
            summed_rewards = torch.tensor(summed_rewards)
            next_observations = next_observations[n_steps]
            experience = (observations, actions, summed_rewards, next_observations)

        self.buffer.append(experience)

    def __len__(self):
        return len(self.buffer)
