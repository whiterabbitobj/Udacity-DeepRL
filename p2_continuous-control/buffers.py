from collections import deque, namedtuple
import random

class ReplayBuffer:
    """
    Standard replay buffer to hold memories for later learning. Returns
    random experiences with no consideration of their usefulness.

    When using an agent with a ROLLOUT trajectory, each experience tuple contains
    stacked data of height=ROLLOUT. For instance, STATES would hold 5 states
    stacked instead of the standard single state. Rollout improves learning by
    providing a better estimation of expected reward.
    """
    def __init__(self, buffersize=50000):
        self.buffer = deque(maxlen=buffersize)
        self.experience = namedtuple("experience", field_names=['states','actions','rewards','next_states'])

    def store(self, state, action, reward, next_state):
        experience = self.experience(state, action, reward, next_state)
        self.buffer.append(experience)

    def sample(self, batchsize):
        batch = random.sample(self.buffer, k=batchsize)
        return  self.experience(*zip(*batch))

    def __len__(self):
        return len(self.buffer)

class nStepBuffer:
    """
    Holds frames for rollout length stacking.
    """
    def __init__(self, size=5):
        self.buffer = deque(maxlen=size)
        self.experience = namedtuple("experience", field_names=['state','action','reward','next_state'])

    def store(self, frame):
        self.buffer.append(frame)

    def __len__(self):
        return len(self.buffer)
    #
    # def __iter__(self):
    #     return self
    #
    # def __next__(self):
    #     return self