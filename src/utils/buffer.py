import random
from collections import deque

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity):
        """
        Initialize the replay buffer.

        Args:
            capacity (int): Maximum size of the buffer.
        """
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        """
        Add a transition to the buffer.

        Args:
            state (np.ndarray): Current state.
            action (int): Action taken.
            reward (float): Reward received.
            next_state (np.ndarray): Next state.
            done (bool): Whether the episode has ended.
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        Sample a batch of transitions from the buffer.

        Args:
            batch_size (int): Number of transitions to sample.

        Returns:
            tuple of np.ndarray: A tuple containing states, actions, rewards, next_states, and dones.
        """
        # Batch: list of tuples (state, action, reward, next_state, done)
        batch = random.sample(self.buffer, batch_size)

        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return states, actions, rewards, next_states, dones

    def reset(self):
        """
        Clear the buffer.
        """
        self.buffer.clear()

    def __len__(self):
        """
        Get the current size of the buffer.

        Returns:
            int: Current size of the buffer.
        """
        return len(self.buffer)
