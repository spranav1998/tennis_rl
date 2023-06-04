# Cooperative Multi-Agent Reinforcement Learning for a Tennis Environment
By: Pranav

In this project, I delved into the exciting field of Multi-Agent Deep Reinforcement Learning (MADRL), applying a Multi-Agent Deep Deterministic Policy Gradient (MADDPG) algorithm to train two agents to cooperate in a Unity-generated virtual tennis environment.

The agents were required to control rackets and bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play. The task is considered solved when the agents get an average score of +0.5 over 100 consecutive episodes.

The MADDPG algorithm, an extension of Deep Deterministic Policy Gradient (DDPG) for multi-agent scenarios, is a method that uses actor-critic architecture while handling challenges of multi-agent environments such as the changing policy of other agents. In this scenario, each agent had its actor and critic network, which consists of fully-connected layers with ReLU activation functions and final layer weights initialized with uniform noise to ensure exploration. 

The `Actor` network maps states to actions, using two fully connected layers with 100 and 50 neurons respectively. On the other hand, the `Critic` network, takes in states and actions of all agents and outputs the estimated Q-values. The critic network also uses two fully connected layers with 300 neurons each. 

Furthermore, I implemented an experience replay buffer to stabilize the learning process by allowing the agent to learn from past experiences. The use of a separate target network was also employed to introduce a delay in updating the weights, a technique that is crucial for stable learning due to the inherent instability of learning from correlated experiences.

The key hyperparameters used for training were a buffer size of 100,000 experiences, and a mini-batch size of 128. The Q-networks were updated every two steps, and the learning rate for both the actor and critic networks was set to 0.001.

Applying these techniques, the agents were able to solve the environment in just over 4839 episodes, effectively demonstrating the application and utility of MADDPG in training multiple agents to accomplish cooperative tasks in high-dimensional state spaces.

Here is the plot:
![plot](https://github.com/spranav1998/tennis_rl/blob/994c60c6aa4363bdb0b6480f15db9d25286941af/plot/training_metrics.jpg)
