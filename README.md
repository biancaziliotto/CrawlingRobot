# CrawlingRobot

Course 'Neural Networks and Deep Learning' - Project 11 \
_Use Q-learning to train an agent to control a crawling robot that has to learn how to move a 2-link finger to move forward. A positive reward is generated when moving forward and a negative reward is generated when moving backward or not moving_.

## Physical system

The physical system is simulated in MuJoCo (Multi-Joint dynamics with Contact). It consists of a two-link articulated mechanism anchored to a free-floating base. The base and links are connected through hinge joints, allowing for planar motion in a vertical plane. The system is designed to simulate interactions with the environment, including collisions and contact forces. Each joint is actuated by a torque motor, controlled via input signals.

## Train policy

The model is trained with Q-Learning. The reward function combines:
- A position reward term, positive for forward movement and negative for backward movement.
- An upright reward term, penalizing rotations of the base.\
The episode length varies within a defined range: episodes are terminated earlier if the average speed is smaller than a minimum threshold.

## Run and evaluate policy

The trained policy achieves an average speed of 0.5 m/s.

https://github.com/user-attachments/assets/db03962c-0a66-4114-873b-ae08e8d68e18

Here we plot the recorded joint angles in the first 2 seconds of motion, after random initialization of the robot position. Notice that the gait cycle duration is smaller than 0.5 s.

![image](https://github.com/user-attachments/assets/a31d5e4d-b4c6-4159-80a3-f43687d429a2)


