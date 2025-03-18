import wandb

from env import Env
from utils.q_net import QNet


class Agent:
    def __init__(self, cfg):
        """
        Initialize agent.
        """
        self.cfg = cfg
        self.env = Env(cfg)
        self.input_size = self.env.state_dim + self.env.action_dim
        self.Q_net = QNet(self.input_size, 1, cfg)

        print("Agent initialized.")
        return

    def train(self):
        """
        - Sample from environment.
        - Update Qnet.
        """

        # wandb.log(
        #     data={
        #         "avg_episode_reward": avg_episode_reward,
        #         "eps_len": avg_episode_len,
        #           ...
        #     },
        #     step=self.epoch,
        # )

        pass

    def eval(self):
        """
        - Evaluate Qnet.
        """
        pass
