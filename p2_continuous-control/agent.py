import numpy as np
import copy
# import random
from collections import namedtuple, deque
from models import ActorNet, CriticNet
#
import torch
import torch.nn as nn
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
        """
        Implementation of D4PG:
        "Distributed Distributional Deterministic Policy Gradients"
        As described in the paper at: https://arxiv.org/pdf/1804.08617.pdf

        Much thanks also to the original DDPG paper:
        https://arxiv.org/pdf/1509.02971.pdf

        And to the work of Bellemare et al:
        "A Distributional Perspective on Reinforcement Learning"
        https://arxiv.org/pdf/1707.06887.pdf

        D4PG utilizes distributional value estimation, n-step returns,
        prioritized experience replay (PER), distributed K-actor exploration,
        and off-policy actor-critic learning to achieve very fast and stable
        learning for continuous control tasks.

        This version of the Agent is written to interact with Udacity's
        Continuous Control robotic arm manipulation environment which provides
        20 simultaneous actors, negating the need for K-actor implementation.
        Thus, this code has no multiprocessing functionality.

        In the original D4PG paper, it is suggested in the data that PER does
        not have significant (or perhaps any at all) effect on the speed or
        stability of learning. Thus, it too has been left out of this
        implementation but may be added as a future TODO item.
        """

        self.t_step = 0
        self.state_size = state_size
        self.action_size = action_size
        self.agent_count = agent_count
        self.rollout = rollout
        self.gamma = gamma
        self.C = C
        self.e = .3
        self.e_decay = 0.9999

        # Set up memory buffers, one for Replay Buffer and one to handle
        # collecting data for n-step returns
        self.memory = ReplayBuffer()
        self.reset_n_step_memory()

        self.actor = ActorNet(state_size, action_size)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=a_lr)

        self.critic = CriticNet(state_size, action_size)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=c_lr)
        self.critic_loss = nn.CrossEntropyLoss()


    def reset_nstep_memory(self):
        """
        Creates (or recreates to zero an existing) deque to handle nstep returns.
        """
        self.n_step_memory = deque(maxlen=self.rollout)

    def initialize_memory(self, pretrain_length):
        """
        Fills up the ReplayBuffer memory with PRETRAIN_LENGTH number of experiences
        before training begins.
        """
        states = env.states
        while len(self.memory) < pretrain_length:
            actions = np.random.uniform(-1, 1, (self.agent_count, self.action_size))
            next_states, rewards, dones = env.step(actions)
            self.step(state, action, reward, next_state, pretrain=True)
            states = next_states

    def act(self, states):
        actions = self.actor(states).detach().numpy()
        noise = self._gauss_noise(actions.shape)
        actions += noise
        actions = np.clip(actions, -1, 1)
        return actions

    def step(self, states, actions, rewards, next_states, pretrain=False):
        # Current SARS' tuple is used with last ROLLOUT steps to generate a new
        # memory
        agent._store_memories(zip(states, actions, rewards, next_states))

        if pretrain:
            return

        agent.learn()
        self.t_step += 1
        self.e *= self.e_decay

    def _store_memories(self, experiences):
        """
        Once the n_step_memory holds ROLLOUT number of sars' tuples, then a full
        memory can be added to the ReplayBuffer.
        """
        self.n_step_memory.append(experiences)

        # If ROLLOUT steps haven't been taken in a new episode, then don't store
        # a SARS' tuple yet
        if len(self.n_step_memory) < self.rollout:
            return

        # Unpacks and stores the SARS' tuple for each actor in the environment
        # thus, each timestep actually adds K_ACTORS memories to the buffer,
        # for the Udacity environment this means 20 memories each timestep.
        for actor in zip(*self.n_step_memory):
            states, actions, rewards, next_states = zip(*actor)
            n_steps = self.rollout - 1
            rewards = np.fromiter((rewards[i] * gamma**i for i in range(n_steps)), float, count=n_steps)
            rewards = rewards.sum()
            # store the current state, current action, cumulative discounted
            # reward from t -> t+n-1, and the next_state at t+n (S't+n)
            self.memory.store(states[0], actions[0], rewards, next_states[-1])

    def learn(self):
        batch = self.memory.sample()
        states, actions, rewards, next_states = batch

        # states.to
        # next_states.to(self.device)

        # Calculate Yᵢ from target networks using θ' and W'
        target = self._get_targets(rewards, next_states)
        # Calculate value distribution for current state using weights W
        current_value_dist = self.critic(states, actions)
        # Predict action for actor network loss calculation using θ
        current_action_prediction = self.actor(states)
        new_value_dist = self.critic(states, current_action_prediction)

        # Calculate LOSSES
        critic_loss = self.critic_loss(target, current_value_dist)
        actor_loss = (current_action_prediction * new_value_dist).mean()

        # Perform gradient descent
        self.actor_optim.zero_grad()
        self.critic_optim.zero_grad()

        actor_loss.backward()
        critic_loss.backward()

        self.actor_optim.step()
        self.critic_optim.step()

        ### Soft-update like in original DDPG
        #self._soft_update(self.critic_target, self.critic, self.tau)
        #self._soft_update(self.actor_target, self.actor, self.tau)

        ### Hard update as in DQN and implied by D4PG paper
        if  self.t_step % self.C == 0:
            self.critic_target.load_state_dict(self.critic.state_dict())
            self.actor_target.load_state_dict(self.critic.state_dict())

    def _categorical(self,
                    rewards,
                    probs,
                    vmin = -10,
                    vmax = 10,
                    num_atoms = 51,
                    gamma = self.gamma):
        """
        Returns the projected value distribution for the input state/action pair
        """

        rewards = torch.tensor(rewards).unsqueeze(-1)
        atoms = torch.linspace(vmin, vmax, num_atoms)
        delta_z = (vmax - vmin) / (num_atoms - 1)

        projected_atoms = rewards + gamma * atoms.view(1,-1)
        b = (projected_atoms - vmin) / delta_z

        lower_bound = b.floor()
        upper_bound = b.ceil()

        m_lower = (upper_bound + (lower_bound == upper_bound).float() - b) * probs
        m_upper = (b - lower_bound) * probs

        projected_probs = torch.tensor(np.zeros(probs.size))
        for idx in range(probs.size(0)):
            projected_probs[idx].index_add_(0, lower_bound[idx], m_lower[idx].double())
            projected_probs[idx].index_add_(0, upper_bound[idx], m_lower[idx].double())

        return projected_probs

    def _soft_update(self, target, active):
        for t_param, param in zip(target.parameters(), active.parameters()):
            t_param.data.copy_(tau*param.data + (1-tau)*t_param.data)

    def _get_target(self, rewards, next_states):
        target_actions = self.actor_target(next_states).detach()
        target_probs = self.critic_target(next_states, target_actions).detach()
        projected_probs = self._categorical(rewards, target_probs, gamma=self.gamma**self.rollout)
        return rewards + projected_probs

    def _gauss_noise(self, shape):
        n = np.random.normal(0, 1, shape)
        return self.e*n
