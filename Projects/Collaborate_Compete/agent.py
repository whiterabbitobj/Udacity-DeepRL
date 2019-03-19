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
    def __init__(self, env, args, agent_count,
                e_decay = 1,
                e_min = 0.05):
        """
        Initialize a MAD4PG network.
        """
        # self.device = args.device
        self.update_type = "hard"
        self.framework = "MAD4PG"
        self.t_step = 0
        self.episode = 0
        self.C = args.C
        self._e = args.e
        self.e_min = e_min
        self.e_decay = e_decay
        # self.gamma = args.gamma
        self.state_size = env.state_size
        self.action_size = env.action_size

        self.agent_count = agent_count
        self.agents = [D4PG_Agent(self.state_size, self.action_size, args, self.agent_count) for _ in range(self.agent_count)]
        self.batch_size = args.batch_size
        # self.buffer_size = args.buffer_size
        # Set up memory buffers, currently only standard replay is implemented #
        self.memory = ReplayBuffer(args.device, args.buffer_size, args.gamma, args.rollout)

        self.new_episode()
        for agent in self.agents:
            self._update_networks(agent, force_hard=True)

    def act(self, observations, eval=False):
        """
        For each agent in the MAD4PG network, choose an action from the ACTOR
        """

        assert len(observations) == len(self.agents), "Num OBSERVATIONS does not match num AGENTS."
        actions = np.array([agent.act(obs) for agent, obs in zip(self.agents, observations)])
        if not eval:
            noise = self._gauss_noise(actions.shape)
            actions += noise
        return np.clip(actions, -1, 1)

    def step(self, observations, actions, rewards, next_observations, dones, pretrain=False):
        """
        Add the current SARS' tuple into the short term memory, then learn
        """

        # Current SARS' stored in short term memory, then stacked for NStep
        #print("len of:\no: {}, a: {}, r: {}, no: {}, d: {}".format(len(observations), len(actions), len(rewards), len(next_observations), len(dones)))
        experience = (observations, actions, rewards, next_observations, dones)
        self.memory.store_trajectory(experience)
        self.t_step += 1

        # Don't learn if pretraining is being executed
        if not pretrain:
            self.learn()

    def learn(self):
        """
        Perform a learning step on all agents in the network.
        """

        # Sample from replay buffer, REWARDS are sum of (ROLLOUT - 1) timesteps
        # Already calculated before storing in the replay buffer.
        # NEXT_OBSERVATIONS are ROLLOUT steps ahead of OBSERVATIONS
        for a_num, agent in enumerate(self.agents):
            batch = self.memory.sample(self.batch_size)
            observations, actions, rewards, next_observations, dones = batch
            agent.learn(observations[a_num], actions, rewards[a_num], next_observations[a_num], dones)
            self._update_networks(agent)

    def initialize_memory(self, pretrain_length, env):
        """
        Fills up the ReplayBuffer memory with PRETRAIN_LENGTH number of experiences
        before training begins.
        """

        if len(self.memory) >= pretrain_length:
            print("Memory already filled, length: {}".format(len(self.memory)))
            return

        print("Initializing memory buffer.")
        observations = env.states
        while len(self.memory) < pretrain_length:
            actions = np.random.uniform(-1, 1, (self.agent_count, self.action_size))
            next_observations, rewards, dones = env.step(actions)
            self.step(observations, actions, rewards, next_observations, dones, pretrain=True)
            if self.t_step % 10 == 0 or len(self.memory) >= pretrain_length:
                print("Taking pretrain step {}... memory filled: {}/{}\
                    ".format(self.t_step, len(self.memory), pretrain_length))

            observations = next_observations
        print("Done!")
        self.t_step = 0

    def new_episode(self):
        """
        Handle any cleanup or steps to begin a new episode of training.
        """

        self.memory.init_n_step()
        self.episode += 1

    def _update_networks(self, agent, force_hard=False):
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
        # self.agent_count = env.agent_count
        self.actor_learn_rate = args.actor_learn_rate
        self.critic_learn_rate = args.critic_learn_rate
        # self.action_size = env.action_size
        # self.state_size = env.state_size
        # self.C = args.C
        self.gamma = args.gamma
        self.rollout = args.rollout
        self.tau = args.tau
        #self.update_type = update_type

        self.num_atoms = args.num_atoms
        self.vmin = args.vmin
        self.vmax = args.vmax
        self.atoms = torch.linspace(self.vmin, self.vmax, self.num_atoms).to(self.device)

        #                    Initialize ACTOR networks                         #
        self.actor = ActorNet(state_size, action_size).to(self.device)
        self.actor_target = ActorNet(state_size, action_size).to(self.device)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.actor_learn_rate, weight_decay=l2_decay)
        #                   Initialize CRITIC networks                         #
        self.critic = CriticNet(state_size, action_size * agent_count, self.num_atoms).to(self.device)
        self.critic_target = CriticNet(state_size, action_size * agent_count, self.num_atoms).to(self.device)
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

    def learn(self, obs, actions, rewards, next_obs, dones):
        """
        Performs a distributional Actor/Critic calculation and update.
        Actor πθ and πθ'
        Critic Zw and Zw' (categorical distribution)
        """

        # Calculate Yᵢ from target networks using πθ' and Zw'
        # These tensors are not needed for backpropogation, so detach from the
        # calculation graph (literally doubles runtime if this is not detached)
        target_dist = self._get_targets(rewards, next_obs, dones).detach()

        # Calculate log probability DISTRIBUTION using Zw w.r.t. stored actions
        log_probs = self.critic(obs, actions, log=True)

        # Calculate the critic network LOSS (Cross Entropy), CE-loss is ideal
        # for categorical value distributions as utilized in D4PG.
        # estimates distance between target and projected values
        critic_loss = -(target_dist * log_probs).sum(-1).mean()


        # Predict action for actor network loss calculation using πθ
        predicted_action = self.actor(obs)

        # Predict value DISTRIBUTION using Zw w.r.t. action predicted by πθ
        probs = self.critic(obs, predicted_action)

        # Multiply probabilities by atom values and sum across columns to get
        # Q-Value
        expected_reward = (probs * self.atoms.unsqueeze(0)).sum(-1)

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

        rewards = rewards.unsqueeze(-1)
        delta_z = (vmax - vmin) / (num_atoms - 1)

        # Rewards were stored with 0->(N-1) summed, take Reward and add it to
        # the discounted expected reward at N (ROLLOUT) timesteps
        projected_atoms = rewards + gamma**rollout * atoms.unsqueeze(0) * (1 - dones)
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

    def _get_targets(self, rewards, next_obs, dones):
        """
        Calculate Yᵢ from target networks using πθ' and Zw'
        """
        #print(next_obs.shape)
        target_actions = self.actor_target(next_obs)
        print(target_actions.shape)
        # target_probs = torch.zeros(rewards, self.num_atoms)
        target_probs = self.critic_target(next_obs, target_actions)
        # Project the categorical distribution onto the supports
        projected_probs = self._categorical(rewards, target_probs, dones)
        return projected_probs
