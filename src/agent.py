import torch

from env import Env
from utils.q_net import QNet


class Agent:
    def __init__(self, cfg):
        self.cfg = cfg
        self.env = Env(cfg)
        self.input_size = self.env.state_dim + self.env.action_dim
        self.Q_net = QNet(self.input_size, 1, cfg)

        print("Agent initialized.")

        return
