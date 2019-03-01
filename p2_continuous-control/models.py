import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ActorNet(nn.Module):
    """
    Actor network that approximates the non-linear function π(θ)
    """
    def __init__(self, state_size, action_size, layer_sizes=[400,300]):
        super(ActorNet, self).__init__()

        self.hidden_layers = nn.ModuleList([nn.Linear(state_size, layer_sizes[0])])
        inner_sizes = zip(layer_sizes[:-1], layer_sizes[1:])
        self.hidden_layers.extend([nn.Linear(i, o) for i, o in inner_sizes])
        self.output = nn.Linear(layer_sizes[-1], action_size)

    def forward(self, state):
        """
        Build a network that maps state -> action values.
        """
        x = F.relu(self.hidden_layers[0](state))
        for layer in self.hidden_layers[1:]:
            x = F.relu(layer(x))
        x = self.output(x).tanh()
        return x

class CriticNet(nn.Module):
    """
    Actor network that approximates the non-linear function π(θ)
    """
    def __init__(self, state_size, actions, layer_sizes=[400,300]):
        super(ActorNet, self).__init__()
