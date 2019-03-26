# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from buffers import ReplayBuffer
from models import ActorNet, CriticNet


class MAD4PG_Net:
    """
    This implementation uses a variant of OpenAI's MADDPG:
    https://arxiv.org/abs/1706.02275
    but utilizes D4PG as a base algorithm:
    https://arxiv.org/pdf/1804.08617.pdf
    in what I will call MAD4PG.
    """

    def __init__(self, env, args,
                e_decay = 1,
                e_min = 0.05):
        """
        Initialize a MAD4PG network.
        """

        self.update_type = args.update_type
        self.framework = "MAD4PG"
        self.t_step = 0
        self.episode = 0
        self.C = args.C
        self._e = args.e
        self.e_min = e_min
        self.e_decay = e_decay
        self.tau = args.tau
        self.state_size = env.state_size
        self.action_size = env.action_size

        self.agent_count = env.agent_count
        self.agents = [D4PG_Agent(self.state_size, self.action_size, args, self.agent_count) for _ in range(self.agent_count)]
        self.batch_size = args.batch_size

        # Set up memory buffers, currently only standard replay is implemented
        # For multi-agent training, only a single buffer is needed for all
        # agents to draw upon.
        # For certain environments, such as sparse rewards, PER would be
        # beneficial to implement
        self.memory = ReplayBuffer(args.device, args.buffer_size, args.gamma, args.rollout, self.agent_count)

        self.new_episode()
        for agent in self.agents:
            self.update_networks(agent, force_hard=True)

    def act(self, obs, eval=False):
        """
        For each agent in the MAD4PG network, choose an action from the ACTOR
        """

        assert len(obs) == len(self.agents), "Num OBSERVATIONS does not match num AGENTS."
        actions = np.array([agent.act(o) for agent, o in zip(self.agents, obs)])
        if not eval:
            actions += self._gauss_noise(actions.shape)
        return np.clip(actions, -1, 1)

    def store(self, obs, next_obs, actions, rewards, dones):
        """
        Store an experience tuple in the ReplayBuffer
        """
        self.memory.store((obs, next_obs, actions, rewards, dones))

    def learn(self):
        """
        Perform a learning step on all agents in the network.
        """
        self.t_step += 1
        # Sample from replay buffer, REWARDS are sum of (ROLLOUT - 1) timesteps
        # Already calculated before storing in the replay buffer.
        # NEXT_OBSERVATIONS are ROLLOUT steps ahead of OBSERVATIONS
        batch = self.memory.sample(self.batch_size)
        obs, next_obs, actions, rewards, dones = batch

        target_actions = [agent.actor_target(next_obs[i]) for i, agent in enumerate(self.agents)]
        target_actions = torch.cat(target_actions, dim=-1).detach()
        #print(target_actions.shape)
        #print(target_actions[0])
        predicted_actions = [agent.actor(obs[i]) for i, agent in enumerate(self.agents)]
        predicted_actions = torch.cat(predicted_actions, dim=-1).detach()
        # print(obs.shape)
        obs = obs.transpose(1,0).contiguous().view(self.batch_size, -1)
        # print(obs.shape)
        next_obs = next_obs.transpose(1,0).contiguous().view(self.batch_size, -1)
        for i, agent in enumerate(self.agents):
            agent.learn(obs, next_obs, actions, target_actions, predicted_actions, rewards[i], dones[i])
            self.update_networks(agent)

    def initialize_memory(self, pretrain_length, env):
        """
        Fills up the ReplayBuffer memory with PRETRAIN_LENGTH number of experiences
        before training begins.
        """

        if self.memlen >= pretrain_length:
            print("Memory already filled, length: {}".format(len(self.memory)))
            return

        print("Initializing memory buffer.")
        obs = env.states
        while self.memlen < pretrain_length:
            actions = np.random.uniform(-1, 1, (self.agent_count, self.action_size))
            next_obs, rewards, dones = env.step(actions)
            self.store(obs, next_obs, actions, rewards, dones)
            obs = next_obs

            if self.memlen % 25 == 0 or self.memlen >= pretrain_length:
                print("...memory filled: {}/{}".format(self.memlen, pretrain_length))
        print("Done!")

    def new_episode(self):
        """
        Handle any cleanup or steps to begin a new episode of training.
        """

        self.memory.init_n_step()
        self.episode += 1

    def update_networks(self, agent, force_hard=False):
        """
        Updates the network using either DDPG-style soft updates (w/ param TAU),
        or using a DQN/D4PG style hard update every C timesteps.
        """

        if self.update_type == "soft" and not force_hard:
            self._soft_update(agent.actor, agent.actor_target)
            self._soft_update(agent.critic, agent.critic_target)
        elif self.t_step % self.C == 0 or force_hard:
            self._hard_update(agent.actor, agent.actor_target)
            self._hard_update(agent.critic, agent.critic_target)

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

    def _gauss_noise(self, shape):
        """
        Returns the epsilon scaled noise distribution for adding to Actor
        calculated action policy.
        """

        n = np.random.normal(0, 1, shape)
        return self.e*n

    @property
    def e(self):
        """
        This property ensures that the annealing process is run every time that
        E is called.
        Anneals the epsilon rate down to a specified minimum to ensure there is
        always some noisiness to the policy actions. Returns as a property.
        """

        self._e = max(self.e_min, self._e * self.e_decay)
        return self._e

    @property
    def memlen(self):
        return len(self.memory)


class D4PG_Agent:
    """
    D4PG utilizes distributional value estimation, n-step returns,
    prioritized experience replay (PER), distributed K-actor exploration,
    and off-policy actor-critic learning to achieve very fast and stable
    learning for continuous control tasks.

    This version of the Agent is written to interact with Udacity's
    Collaborate/Compete environment featuring two table-tennis playing agents.

    In the original D4PG paper, it is suggested in the data that PER does
    not have significant (or perhaps any at all) effect on the speed or
    stability of learning. Thus, it too has been left out of this
    implementation but may be added as a future TODO item.
    """
    def __init__(self, state_size, action_size, args,
                 agent_count = 1,
                 l2_decay = 0.0001):
        """
        Initialize a D4PG Agent.
        """

        self.device = args.device
        self.framework = "D4PG"
        self.eval = args.eval
        self.actor_learn_rate = args.actor_learn_rate
        self.critic_learn_rate = args.critic_learn_rate

        self.gamma = args.gamma
        self.rollout = args.rollout
        # self.tau = args.tau

        self.num_atoms = args.num_atoms
        self.vmin = args.vmin
        self.vmax = args.vmax
        self.atoms = torch.linspace(self.vmin, self.vmax, self.num_atoms).to(self.device)
        self.atoms = self.atoms.unsqueeze(0)

        #                    Initialize ACTOR networks                         #
        self.actor = ActorNet(state_size, action_size).to(self.device)
        self.actor_target = ActorNet(state_size, action_size).to(self.device)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.actor_learn_rate, weight_decay=l2_decay)
        #                   Initialize CRITIC networks                         #
        self.critic = CriticNet(state_size*agent_count, action_size*agent_count, self.num_atoms).to(self.device)
        self.critic_target = CriticNet(state_size*agent_count, action_size*agent_count, self.num_atoms).to(self.device)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=self.critic_learn_rate, weight_decay=l2_decay)


    def act(self, obs, eval=False):
        """
        Predict an action using a policy/ACTOR network π.
        Scaled noise N (gaussian distribution) is added to all actions todo
        encourage exploration.
        """

        obs = obs.to(self.device)
        with torch.no_grad():
            actions = self.actor(obs).detach().cpu().numpy()
        return actions

    def learn(self, obs, next_obs, actions, target_actions, predicted_actions, rewards, dones):
        """
        Performs a distributional Actor/Critic calculation and update.
        Actor πθ and πθ'
        Critic Zw and Zw' (categorical distribution)
        """

        # Calculate log probability DISTRIBUTION using Zw w.r.t. stored actions
        log_probs = self.critic(obs, actions, log=True)

        # Calculate TARGET distribution/project onto supports (Yi)
        target_probs = self.critic_target(next_obs, target_actions).detach()
        target_dist = self._categorical(rewards, target_probs, dones)

        # Calculate the critic network LOSS (Cross Entropy)
        critic_loss = -(target_dist * log_probs).sum(-1).mean()


        # Predict value DISTRIBUTION using Zw w.r.t. action predicted by πθ
        probs = self.critic(obs, predicted_actions)
        # Multiply probabilities by atom values and sum across columns to get
        # Q-Value
        expected_reward = (probs * self.atoms).sum(-1)
        # Calculate the actor network LOSS (Policy Gradient)
        # Take the mean across the batch and multiply in the negative to
        # perform gradient ascent
        actor_loss = -expected_reward.mean()

        # Perform gradient ascent
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # Perform gradient descent
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        self.actor_loss = actor_loss.item()
        self.critic_loss = critic_loss.item()


    def _categorical(self, rewards, probs, dones):
        """
        Returns the projected value distribution for the input state/action pair

        While there are several very similar implementations of this Categorical
        Projection methodology around github, this function owes the most
        inspiration to Zhang Shangtong and his excellent repository located at:
        https://github.com/ShangtongZhang
        """

        # Create local vars to keep code more concise
        vmin = self.vmin
        vmax = self.vmax
        atoms = self.atoms
        num_atoms = self.num_atoms
        gamma = self.gamma
        rollout = self.rollout

        # rewards/dones shape from [batchsize,] to [batchsize,1]
        rewards = rewards.unsqueeze(-1)
        dones = dones.unsqueeze(-1).type(torch.float)

        delta_z = (vmax - vmin) / (num_atoms - 1)

        # Rewards were stored with 0->(N-1) summed
        # shape [batchsize, num_atoms]
        projected_atoms = rewards + gamma**rollout * atoms * (1 - dones)
        projected_atoms.clamp_(vmin, vmax)
        b = (projected_atoms - vmin) / delta_z

        # It seems that on professional level GPUs (for instance on AWS), the
        # floating point math is accurate to the degree that a tensor printing
        # as 99.00000 might in fact be 99.000000001 in the backend, perhaps due
        # to binary imprecision, but resulting in 99.00000...ceil() evaluating
        # to 100 instead of 99. Forcibly reducing the precision to the minimum
        # seems to be the only solution to this problem, and presents no issues
        # to the accuracy of calculating lower/upper_bound correctly.
        precision = 1
        b = torch.round(b * 10**precision) / 10**precision
        lower_bound = b.floor()
        upper_bound = b.ceil()

        m_lower = (upper_bound + (lower_bound == upper_bound).float() - b) * probs
        m_upper = (b - lower_bound) * probs

        projected_probs = torch.tensor(np.zeros(probs.size())).to(self.device)

        for idx in range(probs.size(0)):
            projected_probs[idx].index_add_(0, lower_bound[idx].long(), m_lower[idx].double())
            projected_probs[idx].index_add_(0, upper_bound[idx].long(), m_upper[idx].double())
        return projected_probs.float()
