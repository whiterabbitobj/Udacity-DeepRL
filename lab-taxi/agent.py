import numpy as np
from collections import defaultdict

### Edited for initial commit testing

class Agent:

    def __init__(self, nA=6, gamma=.9, alpha=.9,
                 epsilon_start=1, epsilon_decay=.999, epsilon_min=0.25):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        - gamma: Discount rate (probably not implemented)
        - alpha: learning-rate multiplier
        - epsilon_start: value epsilon will hold on first episode
        - epsilon_decay: rate at which epsilon approaches nil
        - epsilon_min: minimum value of epsilon to guarantee there will always
            be some exploration over pure-greedy exploitation
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.epsilon = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.gamma = gamma
        self.alpha = alpha
        print("Epsilon: {}, E Decay: {}, E Min: {}, Gamma: {}, Alpha: {}".format(self.epsilon, self.epsilon_decay, self.epsilon_min, self.gamma, self.alpha))

    def get_probs(self, state_values):
        self.epsilon = self.epsilon * self.epsilon_decay
        probs = np.ones(self.nA) * self.epsilon / self.nA
        probs[np.argmax(state_values)] = 1 - self.epsilon + (self.epsilon / self.nA)
        return probs

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """

        if state in self.Q:
            prob = self.get_probs(self.Q[state])
        else:
            prob = np.ones(self.nA) / self.nA
        return np.random.choice(np.arange(self.nA), p = prob)

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        q_value = self.Q[state][action]
        q_value_next = np.max(self.Q[next_state]) if not done else 0
        g = reward + self.gamma * q_value_next - q_value
        self.Q[state][action] = q_value + self.alpha * g
        return
