import torch
import torch.nn as nn
import torch.optim as optim


class QNet(nn.Module):
    def __init__(self, input_size, output_size, cfg):
        super(QNet, self).__init__()
        self.layers = nn.ModuleList()

        self.layers.append(nn.Linear(input_size, cfg.qnet_units[0]))
        self.layers.append(nn.ReLU())

        for l in range(len(cfg.qnet_units) - 1):
            self.layers.append(nn.Linear(cfg.qnet_units[l], cfg.qnet_units[l + 1]))
            self.layers.append(nn.ReLU())

        self.layers.append(nn.Linear(cfg.qnet_units[-1], output_size))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
