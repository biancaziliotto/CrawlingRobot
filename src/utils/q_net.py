import torch
import torch.nn as nn
import torch.optim as optim


class QNet(nn.Module):
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
