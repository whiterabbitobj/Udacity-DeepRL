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
        observations, actions, rewards, next_observations, dones = zip(*batch)

        # print("o: {}\na: {}\nr: {}\nno: {}\nd: {}".format(observations.shape, actions.shape, rewards.shape, next_observations.shape, dones.shape))

        # For actions, stack/transpose to num_agents, batch_size, action_size,
        # then combine the num_agents and action_size to get the "centralized
        # action-value function" input, outputting [centralized, batch_size]
        # for sampled experience.
        # actions = torch.stack(actions).transpose(1,0).long()
        # ashape = actions.shape
        # actions = actions.view(ashape[0]*ashape[-1], -1).to(self.device)
        actions = torch.stack(actions).long()
        ashape = actions.shape
        actions = actions.view(ashape[0], -1).to(self.device)


        # Transpose the num_agents and batch_size, for easy indexing later
        # e.g. from 64 experiences of 2 agents each, to 2 agents with 64
        # experiences each
        observations = torch.stack(observations).transpose(1,0).to(self.device)
        rewards = torch.cat(rewards).transpose(1,0).to(self.device)
        next_observations = torch.stack(next_observations).transpose(1,0).to(self.device)
        # print(next_observations.shape)
        # next_observations = next_observations.transpose(1,0).to(self.device)
        # print(next_observations.shape)
        dones = torch.cat(dones).transpose(1,0).to(self.device)
        #
        # print(actions.shape)
        # print(observations.shape)
        # print(rewards.shape)
        # print(dones.shape)

        batch = (observations, actions, rewards, next_observations, dones)
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
            observations, actions, rewards, next_observations, dones = zip(*self.n_step)
            n_steps = self.rollout - 1

            # Calculate n-step discounted reward
            # If encountering a terminal state (next_observation == None) then sum
            # the rewards only until the terminal state and report back a terminal
            # state for the trajectory tuple.
            summed_rewards = np.zeros(len(rewards[0]))
            for i in range(n_steps):
                summed_rewards += self.gamma**i * np.array(rewards[i])
                if np.any(dones[i]):
                    n_steps = i
                    break

            # print(summed_rewards)
                # else:
                    # r += self.gamma**i * rewards[i]
            # rewards = r

            # store the current state, current action, cumulative discounted
            # reward from t -> t+n-1, and the next_state at t+n (S't+n)
            observations = observations[0]
            actions = torch.from_numpy(actions[0]).type(torch.float)
            summed_rewards = torch.tensor([summed_rewards]).type(torch.float)
            next_observations = next_observations[n_steps]
            dones = torch.tensor([dones[n_steps]])
            experience = (observations, actions, summed_rewards, next_observations, dones)

        self.buffer.append(experience)

    def __len__(self):
        return len(self.buffer)
