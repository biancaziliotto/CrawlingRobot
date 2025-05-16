# CrawlingRobot

Course 'Neural Networks and Deep Learning' - Project 11 \
_Use Q-learning to train an agent to control a crawling robot that has to learn how to move a 2-link finger to move forward. A positive reward is generated when moving forward and a negative reward is generated when moving backward or not moving_.

https://github.com/user-attachments/assets/db03962c-0a66-4114-873b-ae08e8d68e18

## Physical system

The physical system is simulated in MuJoCo (Multi-Joint dynamics with Contact). It consists of a two-link articulated mechanism anchored to a free-floating base. The base and links are connected through hinge joints, allowing for planar motion in a vertical plane. The system is designed to simulate interactions with the environment, including collisions and contact forces. Each joint is actuated by a torque motor, controlled via input signals.

## Run and evaluate policy

The trained policy achieves an average speed of 0.5 m/s. To reproduce these results, run the following commands in the root of the repository:

```bash
poetry install
poetry run mjpython run.py eval=True
```

Here we plot the recorded joint angles in the first 2 seconds of motion, after random initialization of the robot position. Notice that the gait cycle duration is smaller than 0.5 s.

## Train policy

The model is trained with **Deep Q‑Learning (DQN)**.

### Reward function

* **Position reward** – positive for forward movement and negative for backward movement.
* **Upright reward** – penalizes rotations of the base, to avoid the crawler tripping.

Episodes terminate when either the maximum length is reached or the average cumulative distance falls below a minimum threshold.

### Deep Q‑Learning algorithm

1. **Neural Q‑Network**
   A fully‑connected network maps the n‑dimensional observation vector to Q‑values for the discrete torque actions for each joint.

   ```text
   Observations s  →  QNet  →  Q-Values(s,a)
   ```

   Two copies are maintained:

   * **Online network Qθ** – updated every gradient step.
   * **Target network Qθ_target** – synced with the online weights every *K<sub>target</sub>* environment steps.

2. **Experience Replay Buffer**
   Transitions *(s, a, r, s′, done)* are stored in a cyclic buffer (capacity = 100 k).
   Sampling mini‑batches breaks temporal correlations and stabilizes learning.

3. **ε‑greedy Exploration**

   * Random actions with decaying probability to promote exploration in the initial phases of the training.

4. **DQN vs Double DQN update**

   * During learning the one‑step TD target is computed with one of two schemes:

* **DQN**
   * `y = r + γ * max_a' Q_target(s', a') * (1 – done)`
* **Double DQN** –

  1. `a_max = argmax_a' Q_online(s', a')`
  2. `y = r + γ * Q_target(s', a_max) * (1 – done)`

   
### Code reference

The core of the training logic is implemented in [`agent.py`](./agent.py):

```python
while not done:
    # 1. INTERACT WITH ENV
    action = self.select_action(state)
    next_state, reward, done, info = self.env.step(action)

    # 2. STORE TRANSITION
    self.replay_buffer.add(state, action, reward, next_state, done)

    # 3. LEARN (after warm‑up)
    if len(self.replay_buffer) >= self.warmup_steps:
        if self.step_counter % self.update_steps == 0:
            for _ in range(self.num_training_steps):
                self.train_step()

    # 4. TARGET NETWORK SYNC
    if self.step_counter % self.update_target_steps == 0:
        self.update_target_network()

    # 5. UPDATE EXPLORATION
    if self.step_counter % self.epsilon_update_steps == 0:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
```



![image](https://github.com/user-attachments/assets/a31d5e4d-b4c6-4159-80a3-f43687d429a2)


