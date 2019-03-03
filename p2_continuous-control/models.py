import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

def initialize_weights(net, low, high):
    for param in net.parameters():
        param.data.uniform_(low, high)

class ActorNet(nn.Module):
    """
    Actor network that approximates the non-linear function π(θ)
    """
    def __init__(self,
                 state_size,
                 action_size,
                 layer_sizes = [400,300],
                 weight_low = -3e-3,
                 weight_high = 3e-3):
        super(ActorNet, self).__init__()

        #currently errors if user were to provide a custom layer_sizes array
        #with dimensions other than 2x1
        fc1, fc2 = layer_sizes

        self.fc1 = nn.Linear(state_size, fc1)
        self.fc2 = nn.Linear(fc1, fc2)
        self.output = nn.Linear(fc2, action_size)
        initialize_weights(self, weight_low, weight_high)

    def forward(self, state):
        """
        Build a network that maps state -> action values.
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.output(x).tanh()
        return x



class CriticNet(nn.Module):
    """
    Critic network that approximates the Value of the suggested Actions that
    come from the Actor network.

    Utilizes the actions as an input to the second hidden layer in order to
    approximate the continuous control problem.
    """
    def __init__(self,
                 state_size,
                 action_size,
                 n_atoms = 51,
                 layer_sizes = [400,300],
                 weight_low = -3e-3,
                 weight_high = 3e-3,
                 v_min = -1,
                 v_max = 1):
        super(CriticNet, self).__init__()

        #currently errors if user were to provide a custom layer_sizes array
        #with dimensions other than 2x1
        fc1, fc2 = layer_sizes

        self.fc1 = nn.Linear(state_size, fc1)
        self.fc2 = nn.Linear(fc1 + action_size, fc2)
        self.logits = nn.Linear(fc2, n_atoms)
        self.atoms = torch.linspace(v_min, v_max, n_atoms)
        initialize_weights(self, weight_low, weight_high)

    def forward(self, state, actions):
        """
        Build a network that maps state -> action values.
        """
        x = F.relu(self.fc1(state))
        x = torch.cat([x, actions], dim=1)
        x = F.relu(self.fc2(x))
        x = self.logits(x)
        probs = F.softmax(x, dim=-1)
        log_probs = F.log_softmax(x, dim=-1)
        #self.z = F.softmax(x, dim=1)
        #self.Q_val = (y * self.atoms).sum()
        #dist = torch.distributions.Categorical(logits=x)
        #print(dist)
        #x = dist
        return probs, log_probs
