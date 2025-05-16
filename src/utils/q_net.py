import torch
import torch.nn as nn
import torch.optim as optim


class QNet(nn.Module):
    """
    Q-Network for DQN agent.

    This net takes as input the state of the environment and outputs Q-values for each action.

    Args:
        input_size (int): Size of the input state.
        output_size (int): Size of the output action space.
        cfg (dict): Configuration dictionary containing network parameters.
    """

    def __init__(self, input_size, output_size, cfg):
        super(QNet, self).__init__()

        widths = cfg.qnet_units

        layers = []

        in_dim = input_size
        for hidden_dim in widths:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
