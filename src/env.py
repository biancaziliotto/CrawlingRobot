import os
import pickle
import shutil
import time
from typing import Optional

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np
from omegaconf import OmegaConf
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

        # Initialize physical system
        self.mj_model = mujoco.MjModel.from_xml_path(cfg.xml_path)
        self.mj_data = mujoco.MjData(self.mj_model)

        # Initialize the environment
        self.episode_id = 0
        self.deterministic_start = cfg.deterministic_start
        self.reset()

        # Set termination conditions
        self.min_steps = int(cfg.min_steps)
        self.max_steps = int(cfg.max_steps)
        self.min_avg_distance_per_step = cfg.min_avg_distance_per_step

        # Set the action space
        self.ctrl_low = self.mj_model.actuator_ctrlrange[:, 0]
        self.ctrl_high = self.mj_model.actuator_ctrlrange[:, 1]

        self.levels = cfg.action_levels

        self.discretized_action = np.linspace(
            self.ctrl_low[0], self.ctrl_high[0], num=self.levels
        )

        # Input/Output for QNet
        self.state_dim = len(self.compute_observations())
        self.action_dim = self.levels**self.mj_model.nu

        print(f"state_dim = {self.state_dim}")
        print(f"action_dim = {self.action_dim}")

        self.print_sensors()

        print("Environment initialized.")
        return

    def load_physical_system(self, ckpt_dir: str):
        """
        Load the MuJoCo model from the checkpoint directory.

        Args:
            ckpt_dir (str): Path to the checkpoint directory.
        """
        xml_path = os.path.join(ckpt_dir, "env_model.xml")

        # Load the MuJoCo model
        self.mj_model = mujoco.MjModel.from_xml_path(xml_path)
        self.mj_data = mujoco.MjData(self.mj_model)
        self.ctrl_low = self.mj_model.actuator_ctrlrange[:, 0]
        self.ctrl_high = self.mj_model.actuator_ctrlrange[:, 1]

        print(f"Physical system loaded from {ckpt_dir}")
        return

    def save_physical_system(self):
        """
        Save the MuJoCo model to the checkpoint directory.
        """
        ckpt_dir = self.cfg.checkpoint_dir
        os.makedirs(ckpt_dir, exist_ok=True)

        # 1) copy the original XML
        xml_dst = os.path.join(ckpt_dir, "env_model.xml")
        shutil.copy2(self.cfg.xml_path, xml_dst)

        print(f"Mujoco model saved to {ckpt_dir}")

    def print_sensors(self):
        """
        Print one line per sensor in the model.
        """
        print(f"Total sensors: {self.mj_model.nsensor}\n")
        for sid in range(self.mj_model.nsensor):
            sref = self.mj_model.sensor(sid)
            name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_SENSOR, sid)
            s_type = mujoco.mjtSensor(sref.type).name.replace("mjSENS_", "")
            dim = sref.dim
            adr = sref.adr
            print(
                f"[{sid:2d}] {name:<20s} | type: {s_type:<10s} | dim: {dim} | adr: {adr}"
            )

    def reset(self, seed: Optional[int] = None, options=None):
        """
        Initialize episode.
        """
        super().reset(seed=seed, options=options)

        # Initial position of the crawler
        self.mj_data.qpos = np.zeros(len(self.mj_data.qpos))

        if self.deterministic_start:
            self.mj_data.qpos[-2] = 0
            self.mj_data.qpos[-1] = 0
        else:
            self.mj_data.qpos[-2] = -np.random.rand(1) * np.pi / 4
            self.mj_data.qpos[-1] = np.random.rand(1) * np.pi / 4 - np.pi / 8

        # Reset env state
        mujoco.mj_kinematics(self.mj_model, self.mj_data)
        self.previous_xpos = float(self.mj_data.qpos[0])
        self.previous_action = np.zeros(self.mj_model.nu)

        # Reset "previous" variables
        self.curr_step = 0
        self.cum_distance = 0
        self.episode = {
            "initial_qpos": self.mj_data.qpos.copy(),
            "actions": [],
            "qpos": [],
        }
        self.episode_id += 1
        return

    def visualize(self):
        """
        Update the rendering scene.
        """
        curr_step = 0
        self.cum_distance = 0
        self.mj_data.qpos = self.episode["initial_qpos"]
        self.previous_xpos = float(self.mj_data.qpos[0])
        mujoco.mj_kinematics(self.mj_model, self.mj_data)
        with mujoco.viewer.launch_passive(self.mj_model, self.mj_data) as self.viewer:
            self.viewer.sync()
            time.sleep(1)
            while curr_step < len(self.episode["qpos"]):
                action = self.episode["actions"][curr_step]
                self.mj_data.ctrl[:] = action
                qpos = self.episode["qpos"][curr_step]
                self.mj_data.qpos = qpos
                mujoco.mj_kinematics(self.mj_model, self.mj_data)

                obs = self.compute_observations()
                self._get_position_reward()

                self.viewer.sync()
                time.sleep(0.005)
                curr_step += 1
            self.viewer.close()
        return

    def _get_sensordata(self):
        return self.mj_data.sensordata

    def compute_observations(self):
        """
        Compute the observations.

        We want to exclude absolute x and y positions to avoid state aliasing.

        Returns
            - obs (np.ndarray): Observations of the environment.
                Shape: ((n_body_parts-1) * 3 + n_joints * 2 + n_sensors + n_actions,)
        """
        # POSITIONS / ROTATIONS
        # qpos = [ x  y  z   qw qx qy qz   q_joint1  q_joint2 ... ]
        base_pos = self.mj_data.qpos[:3].copy()  # x y z
        base_quat = self.mj_data.qpos[3:7].copy()  # qw qx qy qz
        joint_qpos = self.mj_data.qpos[7:].copy()  # q_joint1 q_joint2 ...

        # VELOCITIES / ANGULAR VELOCITIES
        # qvel = [ vx vy vz  wx wy wz  q'_joint1 q'_joint2 ... ]
        base_vel = self.mj_data.qvel[:3].copy()  # vx vy vz
        base_ang_vel = self.mj_data.qvel[3:6].copy()  # wx wy wz
        joint_qvel = self.mj_data.qvel[6:].copy()  # q'_joint1 q'_joint2 ...

        # Take only the z position of the base
        base_z = base_pos[2:3]

        # Get the sensor data
        sensor_data = self.mj_data.sensordata.copy()

        # Concatenate observations and previous action
        obs = np.concatenate(
            [
                base_z,
                base_quat,
                joint_qpos,
                base_vel,
                base_ang_vel,
                joint_qvel,
                sensor_data,
                self.previous_action,
            ]
        )

        expected_shape = (
            1  # base_z
            + 4  # base_quat
            + (self.mj_model.nq - 7)  # joint_qpos
            + 3  # base_vel
            + 3  # base_ang_vel
            + (self.mj_model.nv - 6)  # joint_qvel
            + self.mj_model.nsensordata  # sensor_data
            + self.mj_model.nu,  # previous action
        )

        assert obs.shape == expected_shape, f"obs shape {obs.shape} != {expected_shape}"
        return obs

    def _get_position_reward(self):
        curr_x = float(self.mj_data.qpos[0].copy())

        # Get the distance traveled since the last step in the desired direction
        distance = curr_x - self.previous_xpos
        self.cum_distance += distance
        self.previous_xpos = curr_x

        # Calculate the reward based on the distance traveled
        if distance > 0:
            rwd = (
                self.cfg.w_pos_rwd * (1 - np.exp(-self.cfg.k_pos_rwd * distance))
                # * self._is_tip_touching()
            )
        else:
            rwd = self.cfg.w_pos_rwd * (
                -1 + np.exp(-self.cfg.k_pos_rwd * abs(distance))
            )

        return rwd

    def _get_upright_reward(self):
        """
        Returns the upright reward based on the robot's orientation.
        The reward is negative if the robot is not upright.

        Attention: Mujoco and scipy use different conventions for quaternions.

        Returns:
            float: Upright reward.
        """
        curr_rot = self.mj_data.qpos[3:7]
        r = R.from_quat(curr_rot, scalar_first=True)

        euler = r.as_euler("xyz", degrees=False)
        return (1 - abs(euler[1] / np.pi)) * self.cfg.w_upright_rwd

    def compute_reward(self, action):
        """ """
        pos_rwd = self._get_position_reward()
        upright_rwd = self._get_upright_reward()

        rwd = pos_rwd + upright_rwd

        rwd_info = {
            "pos_rwd": pos_rwd,
            "upright_rwd": upright_rwd,
            "rwd": rwd,
        }

        return rwd, rwd_info

    def end_episode(self):
        """
        Check if the episode is done.
        The episode is done if:
            - The maximum number of steps is reached
            - The minimum average distance per step is not reached
        """
        done = False
        if (
            self.curr_step > self.min_steps
            and self.cum_distance < self.curr_step * self.min_avg_distance_per_step
        ):
            done = True
            print(f"{self.cum_distance} in {self.curr_step} steps")

        if self.curr_step >= self.max_steps:
            done = True
            print(f"{self.cum_distance} in {self.curr_step} steps")

        return done

    def decode_action(self, idx: int) -> np.ndarray:
        """
        Maps a scalar action index to a discretized action.
        The action index is a number between 0 and action_dim - 1.

        Args:
            idx (int): Action index.

        Returns:
            np.ndarray: Decoded action.
                Shape: (nu,)
        """
        indices = []
        for _ in range(self.mj_model.nu):
            indices.append(idx % self.levels)
            idx //= self.levels
        indices.reverse()

        decoded_action = []
        for index in indices:
            decoded_action.append(self.discretized_action[index])

        return np.array(decoded_action)

    def step(self, action_idx: int) -> tuple:
        """
        Take a step in the environment.
        Args:
            action_idx (int): Action index.

        Returns:
            obs (np.ndarray): Observations of the environment.
                Shape: ((n_body_parts-1) * 3 + n_joints * 2 + n_sensors + n_actions,)
            rwd (float): Reward received after taking the action.
            done (bool): True if the episode is done, False otherwise.
            rwd_dict (dict): Dictionary containing the reward components.
        """
        # Take a step in the environment
        action = self.decode_action(action_idx)
        self.mj_data.ctrl[:] = action
        mujoco.mj_step(self.mj_model, self.mj_data)

        # Compute the observations and reward
        obs = self.compute_observations()
        rwd, rwd_dict = self.compute_reward(action)

        # Set "previous" variables
        self.previous_action = action

        # Check if the episode is done
        self.curr_step += 1
        done = self.end_episode()

        self.episode["actions"].append(action)
        self.episode["qpos"].append(self.mj_data.qpos.copy())

        return obs, rwd, done, rwd_dict
