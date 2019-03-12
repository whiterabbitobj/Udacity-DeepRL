import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from buffers import ReplayBuffer
from models import QNetwork



class DQN_Agent:
    """
    PyTorch Implementation of DQN/DDQN.
    """
    def __init__(self,
                 state_size,
                 action_size,
                 args,
                 agent_count = 1,
                 lr = 0.0005,
                 batch_size = 64,
                 buffer_size = 500000,
                 C = 300,
                 device = "cpu",
                 epsilon = 1,
                 gamma = 0.99,
                 rollout = 5,
                 tau = 0.0005,
                 l2_decay = 0.0001,
                 momentum = 1,
                 update_type = "hard"):
        """
        Initialize a D4PG Agent.
        """

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.framework = args.framework
        self.eval = args.eval
        self.agent_count = agent_count
        self.learn_rate = lr

        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.action_size = action_size
        self.state_size = state_size
        self.C = C
        self.epsilon = epsilon

        self.gamma = gamma
        self.rollout = rollout
        self.tau = tau
        self.update_type = update_type


        self.t_step = 0
        self.episode = 0
        self.seed = 0

        # Set up memory buffers
        if args.prioritized_experience_replay:
            self.memory = PERBuffer(args.buffersize, self.batchsize, self.framestack, self.device, args.alpha, args.beta)
            self.criterion = WeightedLoss()
        else:
            self.memory = ReplayBuffer(self.device, self.buffer_size, self.gamma, self.rollout)

        #                    Initialize Q networks                         #
        self.q = self._make_model(state_size, action_size, args.pixels)
        self.q_target = self._make_model(state_size, action_size, args.pixels)
        self._hard_update(self.q, self.q_target)
        self.q_optimizer = self._set_optimizer(self.q.parameters(), lr=lr, decay=l2_decay, momentum=momentum)


        self.new_episode()

    def _set_optimizer(self, params,lr, decay, momentum,  optimizer="Adam"):
        """
        Sets the optimizer based on command line choice. Defaults to Adam.
        """

        if optimizer == "RMSprop":
            return optim.RMSprop(params, lr=lr, momentum=momentum)
        elif optimizer == "SGD":
            return optim.SGD(params, lr=lr, momentum=momentum)
        else:
            return optim.Adam(params, lr=lr, weight_decay=decay)

    def _make_model(self, state_size, action_size, use_cnn):
        """
        Sets up the network model based on whether state data or pixel data is
        provided.
        """

        if use_cnn:
            return QCNNetwork(state_size, action_size, self.seed).to(self.device)
        else:
            return QNetwork(state_size, action_size, self.seed).to(self.device)

    def act(self, state, eval=False, pretrain=False):
        """
        Select an action using epsilon-greedy π.
        Always use greedy if not training.
        """

        if np.random.random() > self.epsilon or not eval and not pretrain:
            state = state.to(self.device)
            #self.q.eval()
            with torch.no_grad():
                action_values = self.q(state).detach().cpu()
            #self.q.train()
            action = action_values.argmax(dim=1).unsqueeze(0).numpy()
        else:
            action = np.random.randint(self.action_size, size=(1,1))
        return action.astype(np.long)

    def step(self, states, actions, rewards, next_states, pretrain=False):
        """
        Add the current SARS' tuple into the short term memory, then learn
        """

        # Current SARS' stored in short term memory, then stacked for NStep
        # experience = list(zip(states, actions, rewards, next_states))
        experience = (states, actions, rewards, next_states)
        #print(experience, '\n\n\n\n')
        # self._store_memories(memory)
        self.memory.store_experience(experience)
        self.t_step += 1

        # Learn after done pretraining
        if not pretrain:
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
        batch, is_weights, tree_idx = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, terminal_mask = batch

        q_values = torch.zeros(self.batch_size).to(self.device)
        if self.framework == 'DQN':
            # VANILLA DQN: get max predicted Q values for the next states from
            # the target model
            q_values[terminal_mask] = self.q_target(next_states).detach().max(dim=1)[0]

        if self.framework == 'DDQN':
            # DOUBLE DQN: get maximizing ACTION under Q, evaluate actionvalue
            # under the q_target

            # Max valued action under active network
            max_actions = self.q(next_states).detach().argmax(1).unsqueeze(1)
            # Use the active network action to get the value of the stable
            # target network
            q_values[terminal_mask] = self.q_target(next_states).gather(1, max_actions).squeeze(1)

        targets = rewards + (self.gamma * q_values)

        targets = targets.unsqueeze(1)
        values = self.q(states).gather(1, actions)

        #Huber Loss provides better results than MSE
        if is_weights is None:
            loss = F.smooth_l1_loss(values, targets)
        #Compute Huber Loss manually to utilize is_weights with Prioritization
        else:
            loss, td_errors = self.criterion.huber(values, targets, is_weights)
            self.memory.batch_update(tree_idx, td_errors)

        # Perform gradient descent
        self.q_optimizer.zero_grad()
        loss.backward()
        self.q_optimizer.step()

        self._update_networks()
        self.loss = loss.item()


    def initialize_memory(self, pretrain_length, env):
        """
        Fills up the ReplayBuffer memory with PRETRAIN_LENGTH number of experiences
        before training begins.
        """

        if len(self.memory) >= pretrain_length:
            print("Memory already filled, length: {}".format(len(self.memory)))
            return

        print("Initializing memory buffer.")
        state = env.state
        while len(self.memory) < pretrain_length:
            action = self.act(state, pretrain=True)
            next_state, reward, done = env.step(action)
            if done:
                next_state = None
                env.reset()
                state = env.state
                continue

            self.step(state, action, reward, next_state, pretrain=True)
            if self.t_step % 10 == 0 or len(self.memory) >= pretrain_length:
                print("Taking pretrain step {}... memory filled: {}/{}\
                    ".format(self.t_step, len(self.memory), pretrain_length))

            states = next_state
        print("Done!")
        self.t_step = 0

    def new_episode(self):
        """
        Handle any cleanup or steps to begin a new episode of training.
        """

        self.memory.init_n_step()
        self.episode += 1

    def _categorical(self, rewards, probs):
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
        projected_atoms = rewards + gamma**rollout * atoms.unsqueeze(0)
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

    def _gauss_noise(self, shape):
        """
        Returns the epsilon scaled noise distribution for adding to Actor
        calculated action policy.
        """

        n = np.random.normal(0, 1, shape)
        return self.e*n

    def _get_targets(self, rewards, next_states):
        """
        Calculate Yᵢ from target networks using πθ' and Zw'
        """

        target_actions = self.actor_target(next_states)
        target_probs = self.critic_target(next_states, target_actions)
        # Project the categorical distribution onto the supports
        projected_probs = self._categorical(rewards, target_probs)
        return projected_probs

    def _update_networks(self):
        """
        Updates the network using either DDPG-style soft updates (w/ param \
        TAU), or using a DQN/D4PG style hard update every C timesteps.
        """

        if self.update_type == "soft":
            self._soft_update(self.q, self.q_target)
        elif self.t_step % self.C == 0:
            self._hard_update(self.q, self.q_target)

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
