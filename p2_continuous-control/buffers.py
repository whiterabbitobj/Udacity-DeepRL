from collections import deque, namedtuple
import random
import torch

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
    def __init__(self, device, buffer_size=100000):
        self.buffer = deque(maxlen=buffer_size)
        self.device = device
        self.experience = namedtuple("experience", field_names=['state','action','reward','next_state'])

    def store(self, state, action, reward, next_state):
        experience = self.experience(state, action, reward, next_state)
        self.buffer.append(experience)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, k=batch_size)
        states, actions, rewards, next_states = zip(*batch)
        states = torch.cat(states).to(self.device)
        actions = torch.cat(actions).float().to(self.device)
        rewards = torch.cat(rewards).to(self.device)
        next_states = torch.cat(next_states).to(self.device)
        # return  self.experience(*zip(*batch))
        return (states, actions, rewards, next_states)

    def init_n_step(self, length):
        """
        Creates (or recreates to zero an existing) deque to handle nstep returns.
        """
        self.n_step = deque(maxlen=length)


    def __len__(self):
        return len(self.buffer)
