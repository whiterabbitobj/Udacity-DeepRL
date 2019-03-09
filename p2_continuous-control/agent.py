import copy
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from buffers import ReplayBuffer
from models import ActorNet, CriticNet
# from data_handling import Saver

class D4PG_Agent:
    def __init__(self,
                 state_size,
                 action_size,
                 agent_count,
                 a_lr = 1e-3,
                 c_lr = 1e-3,
                 batch_size = 128,
                 buffer_size = 300000,
                 C = 500,
                 device = "cpu",
                 e = 0.3,
                 e_decay = 1, #0.99999,
                 e_min = 0.05,
                 gamma = 0.99,
                 num_atoms = 75,
                 vmin = 0,
                 vmax = 0.2,
                 rollout = 5,
                 tau = 0.0005,
                 l2_decay = 0.0001,
                 update_type = "hard"):
        """
        PyTorch Implementation of D4PG:
        "Distributed Distributional Deterministic Policy Gradients"
        (Barth-Maron, Hoffman, et al., 2018)
        As described in the paper at: https://arxiv.org/pdf/1804.08617.pdf

        Much thanks also to the original DDPG paper:
        "Continuous Control with Deep Reinforcement Learning"
        (Lillicrap, Hunt, et al., 2016)
        https://arxiv.org/pdf/1509.02971.pdf

        And to:
        "A Distributional Perspective on Reinforcement Learning"
        (Bellemare, Dabney, et al., 2017)
        https://arxiv.org/pdf/1707.06887.pdf

        D4PG utilizes distributional value estimation, n-step returns,
        prioritized experience replay (PER), distributed K-actor exploration,
        and off-policy actor-critic learning to achieve very fast and stable
        learning for continuous control tasks.

        This version of the Agent is written to interact with Udacity's
        Continuous Control robotic arm manipulation environment which provides
        20 simultaneous actors, negating the need for K-actor implementation.
        Thus, this code has no multiprocessing functionality. It could be easily
        added as part of the main.py script.

        In the original D4PG paper, it is suggested in the data that PER does
        not have significant (or perhaps any at all) effect on the speed or
        stability of learning. Thus, it too has been left out of this
        implementation but may be added as a future TODO item.
        """
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.framework = "D4PG"

        self.agent_count = agent_count
        self.actor_learn_rate = a_lr
        self.critic_learn_rate = c_lr
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.action_size = action_size
        self.state_size = state_size
        self.C = C
        self._e = e
        self.e_decay = e_decay
        self.e_min = e_min
        self.gamma = gamma
        self.rollout = rollout
        self.tau = tau
        self.update_type = update_type

        self.num_atoms = num_atoms
        self.vmin = vmin
        self.vmax = vmax
        self.atoms = torch.linspace(vmin, vmax, num_atoms).to(self.device)

        self.t_step = 0
        self.episode = 0

        # Set up memory buffers, currently only standard replay is implemented #
        self.memory = ReplayBuffer(self.device, self.buffer_size)
        #self.saver = Saver(self.framework, args.save_dir)

        #                    Initialize ACTOR networks                         #
        self.actor = ActorNet(state_size, action_size).to(self.device)
        self.actor_target = ActorNet(state_size, action_size).to(self.device)
        self._hard_update(self.actor, self.actor_target)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=a_lr, weight_decay=l2_decay)
        #                   Initialize CRITIC networks                         #
        self.critic = CriticNet(state_size, action_size, num_atoms).to(self.device)
        self.critic_target = CriticNet(state_size, action_size, num_atoms).to(self.device)
        self._hard_update(self.actor, self.actor_target)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=c_lr, weight_decay=l2_decay)

        self.new_episode()

    def act(self, states):
        """
        Predict an action using a policy/ACTOR network π.
        Scaled noise N (gaussian distribution) is added to all actions todo
        encourage exploration.
        """

        states = states.to(self.device)
        actions = self.actor(states).detach().cpu().numpy()
        noise = self._gauss_noise(actions.shape)
        actions += noise
        return np.clip(actions, -1, 1)

    def step(self, states, actions, rewards, next_states, pretrain=False):
        """
        Add the current SARS' tuple into the short term memory, then learn
        """

        # Current SARS' stored in short term memory, then stacked for NStep
        memory = list(zip(states, actions, rewards, next_states))
        self._store_memories(memory)
        self.t_step += 1

        # Don't do any learning if the network is initializing the memory
        if pretrain:
            return

        self.learn()

    def learn(self):
        """
        Performs a distributional Actor/Critic calculation and update.
        Actor πθ and πθ'
        Critic Zw and Zw' (categorical distribution)
        """

        # Sample from replay buffer, REWARDS are sum of (ROLLOUT - 1) timesteps
        # Already calculated before storing in the replay buffer.
        # NEXT_STATES are ROLLOUT steps ahead of STATES
        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states = batch
        atoms = self.atoms.unsqueeze(0)
        # Calculate Yᵢ from target networks using πθ' and Zw'
        # These tensors are not needed for backpropogation, so detach from the
        # calculation graph (literally doubles runtime if this is not detached)
        target_dist = self._get_targets(rewards, next_states).detach()
        # Calculate log probability DISTRIBUTION using Zw w.r.t. stored actions
        log_probs = self.critic(states, actions, log=True)
        # Calculate the critic network LOSS (Cross Entropy), CE-loss is ideal
        # for categorical value distributions as utilized in D4PG.
        # estimates distance between target and projected values
        critic_loss = -(target_dist * log_probs).sum(-1).mean()
        #critic_loss = -(target_dist * (log_probs*atoms).sum(-1).mean()


        # Predict action for actor network loss calculation using πθ
        predicted_action = self.actor(states)
        # Predict value DISTRIBUTION using Zw w.r.t. action predicted by πθ
        probs = self.critic(states, predicted_action)
        # Multiply probabilities by atom values and sum across columns to get
        # Q-Value
        expected_reward = (probs * atoms).sum(-1)
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

        self._update_networks()

        self.actor_loss = actor_loss.item()
        self.critic_loss = critic_loss.item()


    def initialize_memory(self, pretrain_length, env):
        """
        Fills up the ReplayBuffer memory with PRETRAIN_LENGTH number of experiences
        before training begins.
        """
        if len(self.memory) >= pretrain_length:
            print("Memory already filled, length: {}".format(len(self.memory)))
            return

        print("Initializing memory buffer.")
        states = env.states
        while len(self.memory) < pretrain_length:
            actions = np.random.uniform(-1, 1, (self.agent_count, self.action_size))
            next_states, rewards, dones = env.step(actions)
            self.step(states, actions, rewards, next_states, pretrain=True)
            if self.t_step % 10 == 0 or len(self.memory) >= pretrain_length:
                print("Taking pretrain step {}... memory filled: {}/{}\
                      ".format(self.t_step, len(self.memory), pretrain_length))

            states = next_states
        print("Done!")
        self.t_step = 0

    def new_episode(self):
        """
        Handle any cleanup or steps to begin a new episode of training.
        """
        self.memory.init_n_step(self.rollout)
        self.episode += 1

    def _categorical(self, rewards, probs):
        """
        Returns the projected value distribution for the input state/action pair
        """

        # Create function local vars to keep code more concise
        vmin = self.vmin
        vmax = self.vmax
        atoms = self.atoms
        num_atoms = self.num_atoms

        rewards = rewards.unsqueeze(-1)
        delta_z = (vmax - vmin) / (num_atoms - 1)

        projected_atoms = rewards + self.gamma**self.rollout * atoms.unsqueeze(0)#.view(1,-1)
        projected_atoms.clamp_(vmin, vmax)
        b = (projected_atoms - vmin) / delta_z

        lower_bound = b.floor()
        upper_bound = b.ceil()

        m_lower = (upper_bound + (lower_bound == upper_bound).float() - b) * probs
        m_upper = (b - lower_bound) * probs

        projected_probs = torch.tensor(np.zeros(probs.size())).to(self.device)
        for idx in range(probs.size(0)):
            projected_probs[idx].index_add_(0, lower_bound[idx].long(), m_lower[idx].double())
            projected_probs[idx].index_add_(0, upper_bound[idx].long(), m_lower[idx].double())
        return projected_probs.float()

    def _get_targets(self, rewards, next_states):
        """
        Calculate Yᵢ from target networks using πθ' and Zw'
        """

        target_actions = self.actor_target(next_states)
        target_probs = self.critic_target(next_states, target_actions)
        # Project the categorical distribution onto the supports
        projected_probs = self._categorical(rewards, target_probs)
        return projected_probs

    def _gauss_noise(self, shape):
        """
        Returns the epsilon scaled noise distribution for adding to Actor
        calculated action policy.
        """

        n = np.random.normal(0, 1, shape)
        return self.e*n

    def _update_networks(self):
        """
        Updates the network using either DDPG-style soft updates (w/ param \
        TAU), or using a DQN/D4PG style hard update every C timesteps.
        """

        if self.update_type == "soft":
            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critic, self.critic_target)
        elif self.t_step % self.C == 0:
            self._hard_update(self.actor, self.actor_target)
            self._hard_update(self.critic, self.critic_target)


    def _soft_update(self, active, target):
        for t_param, param in zip(target.parameters(), active.parameters()):
            t_param.data.copy_(self.tau*param.data + (1-self.tau)*t_param.data)

    def _hard_update(self, active, target):
        target.load_state_dict(active.state_dict())

    def _store_memories(self, experiences):
        """
        Once the n_step memory holds ROLLOUT number of sars' tuples, then a full
        memory can be added to the ReplayBuffer.
        """
        self.memory.n_step.append(experiences)

        # Abort if ROLLOUT steps haven't been taken in a new episode
        if len(self.memory.n_step) < self.rollout:
            return

        # Unpacks and stores the SARS' tuple for each actor in the environment
        # thus, each timestep actually adds K_ACTORS memories to the buffer,
        # for the Udacity environment this means 20 memories each timestep.
        # for actor in zip(*self.n_step_memory):
        for actor in zip(*self.memory.n_step):
            states, actions, rewards, next_states = zip(*actor)
            n_steps = self.rollout - 1
            rewards = np.fromiter((rewards[i] * self.gamma**i for i in range(n_steps)), float, count=n_steps)
            rewards = rewards.sum()
            # store the current state, current action, cumulative discounted
            # reward from t -> t+n-1, and the next_state at t+n (S't+n)
            states = states[0].unsqueeze(0)
            actions = torch.from_numpy(actions[0]).unsqueeze(0).double()
            rewards = torch.tensor([rewards])
            next_states = next_states[-1].unsqueeze(0)
            self.memory.store(states, actions, rewards, next_states)

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
