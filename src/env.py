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

        self.discretized_action = np.arange(-1, 1.1, 0.1)
        self.action_dim = len(self.discretized_action) ** self.mj_model.nu
        self.state_dim = len(self.compute_observations())
        self.min_steps = 100
        self.max_steps = 20000

        print("Environment initialized.")
        print(f"state_dim = {self.state_dim}")
        print(f"action_dim = {self.action_dim}")

        self.episode_id = 0
        self.reset()
        # self._run_simulation()

        return

    def reset(self, seed: Optional[int] = None, options=None):
        """
        Initialize episode.
        """
        super().reset(seed=seed, options=options)
        self.mj_data.qpos = np.zeros(len(self.mj_data.qpos))
        mujoco.mj_kinematics(self.mj_model, self.mj_data)
        self.previous_pose = self._get_xpos()[1][0].copy()
        self.curr_step = 0
        self.cum_distance = 0
        self.episode = {"initial_pqos": self.mj_data.qpos, "actions": []}
        self.episode_id += 1
        return

    def visualize(self):
        """
        Update the rendering scene.
        """
        curr_step = 0
        self.mj_data.qpos = self.episode["initial_pqos"]
        mujoco.mj_kinematics(self.mj_model, self.mj_data)
        try:
            with mujoco.viewer.launch_passive(
                self.mj_model, self.mj_data
            ) as self.viewer:
                while curr_step < len(self.episode["actions"]):
                    action = self.episode["actions"][curr_step]
                    self.mj_data.ctrl[:] = action
                    mujoco.mj_step(self.mj_model, self.mj_data)
                    self.viewer.sync()
                    time.sleep(0.01)
                    curr_step += 1
                self.viewer.close()
        except RuntimeError as e:
            while curr_step < len(self.episode["actions"]):
                action = self.episode["actions"][curr_step]
                self.mj_data.ctrl[:] = action
                mujoco.mj_step(self.mj_model, self.mj_data)
                self.viewer.sync()
                time.sleep(0.01)
                curr_step += 1
            self.viewer.close()
        return

    def _get_sensordata(self):
        return self.mj_data.sensordata

    def _get_qpos(self):
        return self.mj_data.qpos

    def _get_xpos(self):
        return self.mj_data.xpos

    def _compute_reset(self):
        """
        Check conditions to truncate episode.
        """
        pass

    def compute_observations(self):
        """
        Returns
        """
        observations = []
        observations.extend(self._get_xpos().flatten())
        observations.extend(self._get_qpos().flatten())
        observations.extend(self._get_sensordata().flatten())

        return observations

    def _get_position_reward(self):
        curr_pos = self._get_xpos()[1][0]
        distance = curr_pos - self.previous_pose
        self.cum_distance += distance
        if distance > 0:
            return self.cfg.w_pos_rwd * (1 - np.exp(-self.cfg.k_pos_rwd * distance))
        else:
            return 0

    def _get_energy_reward(self, action):
        return self.cfg.w_energy_rwd * np.exp(
            -self.cfg.k_energy_rwd * np.linalg.norm(action)
        )

    def compute_reward(self, action):
        """
        Returns a scalar value
        Positive reward: the robot moved forward
        Negative reward: the robot moved backward
        """
        pos_rwd = self._get_position_reward()
        energy_rwd = self._get_energy_reward(action)
        rwd = pos_rwd + energy_rwd
        return rwd, {"pos_rwd": pos_rwd, "energy_rwd": energy_rwd, "rwd": rwd}

    def end_episode(self):
        done = False
        if (
            self.curr_step > self.min_steps
            and self.cum_distance < self.curr_step * 0.05
        ):
            done = True
        if self.curr_step >= self.max_steps:
            done = True

        return done

    def step(self, action):
        """
        Execute action and update state.
        Returns reward and observations of next state.
        """

        def decode_action(action):
            levels = len(self.discretized_action)
            actuators = self.mj_model.nu

            # print(action)
            indices = []
            for _ in range(actuators):
                indices.append(action % levels)
                action //= levels
            indices = indices[::-1].copy()

            decoded_action = []
            for index in indices:
                decoded_action.append(self.discretized_action[index])
            # print(decoded_action)

            return np.array(decoded_action)

        action = decode_action(action)
        # print(action)
        self.mj_data.ctrl[:] = action
        mujoco.mj_step(self.mj_model, self.mj_data)
        # print(self.mj_data.qpos)
        self.curr_step += 1
        obs = self.compute_observations()
        rwd, rwd_dict = self.compute_reward(action)
        done = self.end_episode()

        self.episode["actions"].append(action)

        return obs, rwd, done, rwd_dict
