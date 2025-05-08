import time
from typing import Optional

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation as R


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
        self.mode = self.cfg.mode
        self.discretized_action = np.arange(-0.99, 1, 0.11)
        self.action_dim = len(self.discretized_action) ** self.mj_model.nu
        self.episode_id = 0
        self.reset()
        self.state_dim = len(self.compute_observations())
        self.min_steps = 1000
        self.max_steps = 100000

        print("Environment initialized.")
        print(f"state_dim = {self.state_dim}")
        print(f"action_dim = {self.action_dim}")

        # self._run_simulation()

        return

    def reset(self, seed: Optional[int] = None, options=None):
        """
        Initialize episode.
        """
        super().reset(seed=seed, options=options)
        self.mj_data.qpos = np.zeros(len(self.mj_data.qpos))
        self.mj_data.qpos[-2] = -np.random.rand(1) * 1.57 / 2
        self.mj_data.qpos[-1] = np.random.rand(1) * 1.57 / 2 - 1.57 / 4
        self.mj_data.xpos[0] = 0
        mujoco.mj_kinematics(self.mj_model, self.mj_data)
        self.previous_pose = self._get_xpos()[1][0].copy()
        self.previous_action = [0, 0]
        self.curr_step = 0
        self.cum_distance = 0
        self.episode = {"initial_pqos": self.mj_data.qpos.copy(), "actions": []}
        self.episode_id += 1
        return

    def visualize(self):
        """
        Update the rendering scene.
        """
        curr_step = 0
        self.mj_data.qpos = self.episode["initial_pqos"]
        mujoco.mj_kinematics(self.mj_model, self.mj_data)

        with mujoco.viewer.launch_passive(self.mj_model, self.mj_data) as self.viewer:
            self.viewer.sync()
            time.sleep(1)
            while curr_step < len(self.episode["actions"]):
                action = self.episode["actions"][curr_step]
                self.mj_data.ctrl[:] = action
                mujoco.mj_step(self.mj_model, self.mj_data)

                self._get_upright_reward()
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
        observations.extend(self.previous_action)

        return observations

    def _get_position_reward(self):
        curr_pos = self._get_xpos()[1][0]
        distance = curr_pos - self.previous_pose
        self.previous_pose = curr_pos.copy()
        self.cum_distance += distance
        # print(f"distance {distance}")
        if (distance > 0 and self.mode == "forward") or (
            distance < 0 and self.mode == "backward"
        ):
            rwd = self.cfg.w_pos_rwd * (1 - np.exp(-self.cfg.k_pos_rwd * abs(distance)))
        else:
            rwd = self.cfg.w_pos_rwd * (
                -1 + np.exp(-self.cfg.k_pos_rwd * abs(distance))
            )

        # print(f"rwd {rwd}")
        return rwd

    def _get_upright_reward(self):
        curr_rot = self.mj_data.qpos[3:7]
        r = R.from_quat(curr_rot)
        euler = r.as_euler("xyz", degrees=False)
        return -abs(euler[1] / np.pi) * self.cfg.w_upright_rwd

    def _get_energy_reward(self, action):
        return self.cfg.w_energy_rwd * np.exp(
            -self.cfg.k_energy_rwd * np.linalg.norm(action - self.previous_action)
        )

    def compute_reward(self, action):
        """
        Returns a scalar value
        Positive reward: the robot moved forward
        Negative reward: the robot moved backward
        """
        pos_rwd = self._get_position_reward()
        energy_rwd = self._get_energy_reward(action)
        upright_rwd = self._get_upright_reward()
        rwd = pos_rwd + energy_rwd + upright_rwd
        return rwd, {
            "pos_rwd": pos_rwd,
            "energy_rwd": energy_rwd,
            "upright_rwd": upright_rwd,
            "rwd": rwd,
        }

    def end_episode(self):
        done = False
        if (
            self.curr_step > self.min_steps
            and self.cum_distance < self.curr_step * 0.0001
            and self.mode == "forward"
        ):
            done = True
            print(f"{self.cum_distance} in {self.curr_step} steps")
        elif (
            self.curr_step > self.min_steps
            and self.cum_distance > -self.curr_step * 0.0001
            and self.mode == "backward"
        ):
            done = True
            print(self.cum_distance)
            print(f"{self.cum_distance} in {self.curr_step} steps")
        if self.curr_step >= self.max_steps:
            done = True
            print(self.cum_distance)
            print(f"{self.cum_distance} in {self.curr_step} steps")

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
        self.previous_action = action
        done = self.end_episode()

        self.episode["actions"].append(action)

        return obs, rwd, done, rwd_dict
