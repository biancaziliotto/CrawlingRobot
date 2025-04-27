import time
from typing import Optional

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np


class Env(gym.Env):
    """
    Physical system.
    """

    def __init__(self, cfg):
        """
        Initialize physical system.
        """
        self.cfg = cfg
        self.render_mode = "human"
        self.mj_model = mujoco.MjModel.from_xml_path(cfg.xml_path)
        self.mj_data = mujoco.MjData(self.mj_model)

        self.action_dim = self.mj_model.nu
        self.state_dim = self.mj_model.njnt

        print("Environment initialized.")
        print(f"state_dim = {self.state_dim}")
        print(f"action_dim = {self.action_dim}")

        self.reset()
        self._run_simulation()

        return

    def _run_simulation(self, num_steps=1000):
        """
        Update the rendering scene.
        """
        with mujoco.viewer.launch_passive(self.mj_model, self.mj_data) as viewer:
            while viewer.is_running() and self.curr_step < num_steps:
                action = np.random.rand(2)
                self.step(action)
                self.compute_reward(action)
                self.previous_pose = self._get_xpos()[1][0].copy()
                viewer.sync()
                time.sleep(0.01)
        return

    def _get_sensordata(self):
        return self.mj_data.sensordata

    def _get_qpos(self):
        return self.mj_data.qpos

    def _get_xpos(self):
        return self.mj_data.xpos

    def _compute_reward(self):
        """
        Compute reward.
        """
        pass

    def _compute_reset(self):
        """
        Check conditions to truncate episode.
        """
        pass

    def reset(self, seed: Optional[int] = None, options=None):
        """
        Initialize episode.
        """
        super().reset(seed=seed, options=options)
        self.mj_data.qpos[-2:] = [-1.57, 1.57]
        mujoco.mj_kinematics(self.mj_model, self.mj_data)
        self.previous_pose = self._get_xpos()[1][0].copy()
        self.curr_step = 0
        return

    def step(self, action):
        """
        Execute action and update state.
        """
        self.mj_data.ctrl[:] = action
        mujoco.mj_step(self.mj_model, self.mj_data)
        # print(self.mj_data.qpos)
        self.curr_step += 1
        return

    def compute_observations(self):
        """
        Returns
        """
        observations = []
        observations.append(self._get_xpos().flatten())
        observations.append(self._get_qpos().flatten())
        observations.append(self._get_sensordata().flatten())

        return observations

    def _get_position_reward(self):
        curr_pos = self._get_xpos()[1][0]
        distance = curr_pos - self.previous_pose
        if distance > 0:
            return self.cfg.w_pos_rwd * (1 - np.exp(-self.cfg.k_pos_rwd * distance))
        else:
            return 0

    def _get_energy_reward(self, action):
        return self.cfg.w_energy_rwd * np.exp(
            self.cfg.k_energy_rwd * np.linalg.norm(action)
        )

    def compute_reward(self, action):
        """
        Returns a scalar value
        Positive reward: the robot moved forward
        Negative reward: the robot moved backward
        """
        reward = self._get_position_reward() + self._get_energy_reward(action)
        print(reward)
        return reward
