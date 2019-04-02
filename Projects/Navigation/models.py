import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class QNetwork(nn.Module):
    """
    Deep Q-Network Model. Nonlinear estimator for Qπ
    """

    def __init__(self, state_size, action_size, seed, layer_sizes=[64, 64]):
        """
        Initialize parameters and build model.
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.hidden_layers = nn.ModuleList([nn.Linear(state_size, layer_sizes[0])])
        self.output = nn.Linear(layer_sizes[-1], action_size)

        layer_sizes = zip(layer_sizes[:-1], layer_sizes[1:])
        self.hidden_layers.extend([nn.Linear(i, o) for i, o in layer_sizes])


    def forward(self, state):
        """
        Build a network that maps state -> action values.
        """

        x = F.relu(self.hidden_layers[0](state))
        for layer in self.hidden_layers[1:]:
            x = F.relu(layer(x))
        return self.output(x)



# FOR 2D CONVOLUTIONAL NETWORK
class QCNNetwork(nn.Module):
    """
    Deep Q-Network CNN Model for use with learning from pixel data.
    Nonlinear estimator for Qπ
    """
    def __init__(self, state_size, action_size, seed):
        """
        Initialize parameters and build model.
        """

        super(QCNNetwork, self).__init__()
        if len(state_size) == 5:
            _, chans, depth, width, height = state_size
        else:
            _, chans, width, height = state_size
        print("STATESIZE FOR CNN:", state_size)

        outs = [32, 64, 128]
        kernels = [8, 4, 4]
        strides = [4, 2, 2]
        padding = []

        self.conv1 = nn.Conv2d(chans, outs[0], kernels[0], stride=strides[0])
        self.bn1 = nn.BatchNorm2d(outs[0])
        self.conv2 = nn.Conv2d(outs[0], outs[1], kernels[1], stride=strides[1])
        self.bn2 = nn.BatchNorm2d(outs[1])
        self.conv3 = nn.Conv2d(outs[1], outs[2], kernels[2], stride=strides[2])
        self.bn3 = nn.BatchNorm2d(outs[2])

        fc_in = self._get_fc_in(state_size)
        fc_hidden = 512

        self.fc1 = nn.Linear(fc_in, fc_hidden)
        self.fc2 = nn.Linear(fc_hidden, action_size)
        self.seed = torch.manual_seed(seed)

    def _cnn(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        return x

    def _get_fc_in(self, state_size):
        x = torch.rand(state_size)
        x = self._cnn(x)
        fc_in = x.data.view(1, -1).size(1)
        return fc_in

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = self._cnn(state)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# FOR 3D CONVOLUTIONAL KERNELS
# class QCNNetwork(nn.Module):
#     """
#     Deep Q-Network CNN Model for use with learning from pixel data.
#     Nonlinear estimator for Qπ
#     """

#     def __init__(self, state_size, action_size, seed):
#         """
#         Initialize parameters and build model.
#         """
#         super(QCNNetwork, self).__init__()
#         _, chans, depth, width, height = state_size
#
#         print("STATESIZE FOR CNN:", state_size)
#
#         outs = [128, 128*2, 128*2]
#         kernels = [(1,3,3), (1,3,3), (4,3,3)]
#         strides = [(1,3,3), (1,3,3), (1,3,3)]
#         #
#         # outs = [64, 128, 256]
#         # kernels = [(1,4,4), (1,3,3), (4,3,3)]
#         # strides = [(1,2,2), (1,3,3), (1,3,3)]
#         self.conv1 = nn.Conv3d(chans, outs[0], kernels[0], stride=strides[0])
#         self.bn1 = nn.BatchNorm3d(outs[0])
#         self.conv2 = nn.Conv3d(outs[0], outs[1], kernels[1], stride=strides[1])
#         self.bn2 = nn.BatchNorm3d(outs[1])
#         self.conv3 = nn.Conv3d(outs[1], outs[2], kernels[2], stride=strides[2])
#         self.bn3 = nn.BatchNorm3d(outs[2])
#
#         fc_in = self._get_fc_in(state_size)
#         fc_hidden = 512
#
#
#         self.fc1 = nn.Linear(fc_in, fc_hidden)
#         self.fc2 = nn.Linear(fc_hidden, action_size)
#         self.seed = torch.manual_seed(seed)
#
#     def _cnn(self, x):
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = F.relu(self.bn2(self.conv2(x)))
#         x = F.relu(self.bn3(self.conv3(x)))
#         x = x.view(x.size(0), -1)
#         return x
#
#     def _get_fc_in(self, state_size):
#         x = torch.rand(state_size)
#         x = self._cnn(x)
#         fc_in = x.data.view(1, -1).size(1)
#         return fc_in
#
#     def forward(self, state):
#         """Build a network that maps state -> action values."""
#         x = self._cnn(state)
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x



class WeightedLoss(nn.Module):
    """
    Returns Huber Loss with importance sampled weighting.
    """

    def __init__(self):
        super(WeightedLoss, self).__init__()

    def huber(self, values, targets, weights):
        errors = torch.abs(values - targets)
        loss = (errors<1).float()*(errors**2) + (errors>=1).float()*(errors - 0.5)
        weighted_loss = (weights * loss).sum()
        return weighted_loss, errors.detach().cpu().numpy()
