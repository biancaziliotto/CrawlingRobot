import torch
import torch.nn as nn

import wandb
from env import Env
from utils.buffer import ReplayBuffer
from utils.q_net import QNet


class Agent:
    """
    DQN Agent for reinforcement learning.
    This agent interacts with the environment, learns from experiences, and updates its policy.

    Args:
        cfg (dict): Configuration dictionary containing environment and agent parameters.
    """

    def __init__(self, cfg):
        """
        Initialize agent.

        Args:
            cfg (dict): Configuration dictionary containing environment and agent parameters.
        """
        # Initialize the environment
        self.cfg = cfg
        self.env = Env(cfg)

        # Initialize the agent networks
        state_dim = self.env.state_dim
        action_dim = self.env.action_dim

        self.q_network = QNet(state_dim, action_dim, cfg)
        self.target_network = QNet(state_dim, action_dim, cfg)
        self.target_network.load_state_dict(self.q_network.state_dict())

        # Initialize the optimizer and replay buffer
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), cfg.learning.lr)

        self.checkpoint_frequency = cfg.checkpoint_frequency

        # Initialize the epsilon-greedy parameters
        self.epsilon = cfg.learning.epsilon
        self.epsilon_decay = cfg.learning.epsilon_decay
        self.epsilon_update_steps = cfg.learning.epsilon_update_steps
        self.epsilon_min = cfg.learning.epsilon_min
        self.eval_mode = False

        # Initialize the buffer parameters
        self.replay_buffer = ReplayBuffer(capacity=int(cfg.learning.buffer_capacity))
        self.warmup_steps = cfg.learning.warmup_steps

        # Initialize training parameters
        self.batch_size = cfg.learning.batch_size
        self.gamma = cfg.learning.gamma
        self.update_target_steps = cfg.learning.update_target_steps
        self.update_steps = cfg.learning.update_steps
        self.num_training_steps = cfg.learning.num_training_steps

        # Use Double DQN or Vanilla DQN
        self.double_dqn = cfg.learning.double_dqn

        # Counter for updating the target network
        self.step_counter = 1

        print("Agent initialized.")
        return

    def train_step(self):
        """
        Train the agent using a batch of experiences from the replay buffer.

        - Sample a batch of experiences from the replay buffer.
        - Compute the target Q-values using the Double DQN algorithm.
        - Compute the loss between the predicted Q-values and the target Q-values.
        - Perform a gradient descent step to update the Q-network.

        """

        # Sample from the replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        # Shapes out of the replay buffer sampling:
        #   states: (batch_size, state_dim)
        #   actions: (batch_size)
        #   rewards: (batch_size)
        #   next_states: (batch_size, state_dim)
        #   dones: (batch_size)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        # Compute Target Q-values
        with torch.no_grad():
            if self.double_dqn:
                # Double DQN: Use online network to select action, target network to evaluate
                # 1. Use q_network (online) to choose the best next action
                next_actions = self.q_network(next_states).argmax(dim=1, keepdim=True)
                # 2. Use target_network to evaluate that action
                next_q = self.target_network(next_states).gather(1, next_actions)
            else:
                # Vanilla DQN: Use target network to select and evaluate action
                # 1. Use target_network to find the max Q value for the next state
                next_q = self.target_network(next_states).max(dim=1, keepdim=True)[0]

            # Build the target (same for both DQN and Double DQN)
            target_q = rewards + (1 - dones) * self.gamma * next_q

        # Compute the current Q-values and calculate the loss
        current_q = self.q_network(states).gather(1, actions)
        loss = nn.functional.mse_loss(current_q, target_q)

        # Perform a gradient descent step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        wandb.log({"loss": loss.item()}, step=self.step_counter)

    def train(self, num_episodes):
        """
        Train the agent using the environment.

        Algorithm:
            repeat for each episode
                repeat for each env step while not done
                    • act → get (s,a,r,s',done)
                    • store in replay
                    • if replay large enough
                        sample mini-batch(es)
                        gradient step(s)
                    • every K1 env steps: copy online → target
                    • every K2 env steps: update epsilon

        Args:
            num_episodes (int): Number of episodes to train the agent.
        """
        self.env.save_physical_system()
        self.eval_mode = False
        self.q_network.train()

        for episode in range(num_episodes):
            self.episode = episode
            self.env.reset()
            state = self.env.compute_observations()
            done = False
            episode_reward = 0
            episode_pos_reward = 0

            while not done:
                # ---------- 1. INTERACT WITH ENV ----------
                action = self.select_action(state)
                next_state, reward, done, rwd_dict = self.env.step(action)
                wandb.log(rwd_dict, step=self.step_counter)

                # ---------- 2. STORE TRANSITION ----------
                self.replay_buffer.add(state, action, reward, next_state, done)

                # ---------- 3. LEARN (after warmup) ----------
                if len(self.replay_buffer) >= self.warmup_steps:
                    if self.step_counter % self.update_steps == 0:
                        for i in range(self.num_training_steps):
                            self.train_step()

                # ---------- 4. TARGET NETWORK SYNC ----------
                if self.step_counter % self.update_target_steps == 0:
                    self.update_target_network()

                # ---------- 5. UPDATE EXPLORATION ----------
                if (
                    self.step_counter % self.epsilon_update_steps == 0
                    and self.step_counter > self.warmup_steps
                ):
                    self.epsilon = max(
                        self.epsilon_min,
                        self.epsilon * self.epsilon_decay,
                    )

                # ---------- 6. HOUSEKEEPING ----------
                wandb.log(
                    {
                        "epsilon": self.epsilon,
                    },
                    step=self.step_counter,
                )
                state = next_state
                episode_reward += reward
                episode_pos_reward += rwd_dict["pos_rwd"]

                self.step_counter += 1

                if self.step_counter % self.checkpoint_frequency == 0:
                    self.save_model(
                        f"{self.cfg.checkpoint_dir}/model_{int(self.step_counter // self.checkpoint_frequency)}.ckpt"
                    )

            print(f"Episode {episode} finished with reward {episode_reward}")
            # ---------- LOGGING ----------
            wandb.log(
                {
                    "episode_reward": episode_reward,
                    "episode_length": self.env.curr_step,
                    "episode_distance": self.env.cum_distance,
                    "episode_pos_reward": episode_pos_reward,
                    "episode": self.episode,
                }
            )

    def eval(self):
        """
        Set the agent in eval mode.
        """
        self.eval_mode = True
        self.q_network.eval()

    def run_policy(self, num_episodes):
        """
        Function used for evaluation or visualization.

        Args:
            num_episodes (int): Number of episodes to run the policy.
        """
        for episode in range(num_episodes):
            self.env.reset()
            state = self.env.compute_observations()
            done = False
            episode_reward = 0

            while not done:
                action = self.select_action(state)
                next_state, reward, done, rwd_dict = self.env.step(action)

                state = next_state
                episode_reward += reward

                self.step_counter += 1

            print(rwd_dict)
            print(f"reward {episode_reward}")
            self.env.visualize()

    def select_action(self, state):
        """
        Select an action using epsilon-greedy policy.
        If in eval mode, select the action with the highest Q-value.

        Args:
            state (np.ndarray): Current state of the environment.

        Returns:
            int: Action to take.
        """
        if not self.eval_mode and torch.rand(1).item() < self.epsilon:
            return torch.randint(0, self.env.action_dim, (1,)).item()
        with torch.no_grad():
            q_values = self.q_network(torch.FloatTensor(state).unsqueeze(0))
        return q_values.argmax().item()

    def update_target_network(self):
        """
        Update the target network with the weights of the Q-network.
        """
        self.target_network.load_state_dict(self.q_network.state_dict())
        return

    def save_model(self, path):
        """
        Save the Q-network model to a file.

        Args:
            path (str): Path to save the model.
        """
        torch.save(self.q_network.state_dict(), path)
        print(f"Model saved to {path}.")
        return

    def load_model(self, path):
        """
        Load the Q-network model from a file.

        Args:
            path (str): Path to load the model from.
        """
        self.q_network.load_state_dict(torch.load(path))
        print(f"Model loaded from {path}.")
        return
