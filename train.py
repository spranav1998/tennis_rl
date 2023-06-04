# Activate the custom drlnd env as mentioned in the ReadMe.md
# Import the required libraries
from unityagents import UnityEnvironment
import numpy as np
import matplotlib.pyplot as plt
import torch
from collections import deque

# Import the agent and other models
from agent import *

def main():

    # Load the environment
    env = UnityEnvironment(file_name="Tennis_Windows_x86_64/Tennis.exe")

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # number of agents 
    num_agents = len(env_info.agents)
    print('Number of agents:', num_agents)

    # size of each action
    action_size = brain.vector_action_space_size
    print('Size of each action:', action_size)

    # examine the state space 
    states = env_info.vector_observations
    state_size = states.shape[1]
    print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
    print('The state for the first agent looks like:', states[0])

    # Set Random Seed for repeatability
    RANDOM_SEED = 2

    # Define the DDPG Loop
    def ddpg(n_episodes=5000, print_every=100):
        # Load the agent
        num_agents = 2
        multi_agent = MADDPG(num_agents, state_size, action_size, RANDOM_SEED)

        scores_deque = deque(maxlen=print_every)
        scores = []
        max_scores = [] # Maximum score for each episode across all agents
        min_scores = [] # Minimum score for each episode across all agents
        std_scores = [] # Standard deviation of scores for each episode across all agents

        for i_episode in range(1, n_episodes+1):
            env_info = env.reset(train_mode=True)[brain_name]
            states = env_info.vector_observations 
            multi_agent.reset()
            scores_agents = np.zeros(states.shape[0]) 
            t_step = 0
            while True:
                actions = multi_agent.act(states)                       
                env_info = env.step(actions)[brain_name]           
                next_states = env_info.vector_observations      
                rewards = env_info.rewards                      
                dones = env_info.local_done                     
                scores_agents += env_info.rewards                    
                t_step += 1
                multi_agent.step(states, actions, rewards, next_states, dones)
                states = next_states                              
                if np.any(dones):                                 
                    break
            scores_deque.append(np.mean(scores_agents))
            scores.append(np.mean(scores_agents))
            max_scores.append(np.max(scores_agents))
            min_scores.append(np.min(scores_agents))
            std_scores.append(np.std(scores_agents))
            if i_episode % print_every == 0:
                print('\rEpisode {}; Average Score: {:.2f}; Current Score: {:.2f}; Max Score: {:.2f}; Min Score: {:.2f}; Score Std Dev: {:.2f}'.format(
                i_episode, np.mean(scores_deque), np.mean(scores_agents), np.max(scores_agents), np.min(scores_agents), np.std(scores_agents)))

            if i_episode % print_every == 0 or np.mean(scores_deque) > np.max(scores):
                multi_agent.save('tennis_actor', 'tennis_critic')
                
            if np.mean(scores_deque) > 0.5:
                print(f'Evironment Solved in {i_episode} episode')
                multi_agent.save('tennis_actor', 'tennis_critic')
                break

        return scores, max_scores, min_scores, std_scores

    # Run the ddpg loop
    scores, max_scores, min_scores, std_scores = ddpg()

    fig, ((ax1), (ax2), (ax3), (ax4)) = plt.subplots(4, 1, sharex=True, figsize=(15, 12))
    fig.suptitle('Training Metrics')

    ax1.plot(np.arange(1, len(scores)+1), scores)
    ax1.set_ylabel('Score')

    ax2.plot(np.arange(1, len(max_scores)+1), max_scores)
    ax2.set_ylabel('Max Score')

    ax3.plot(np.arange(1, len(min_scores)+1), min_scores)
    ax3.set_ylabel('Min Score')

    ax4.plot(np.arange(1, len(std_scores)+1), std_scores)
    ax4.set_ylabel('Score Std Dev')
    ax4.set_xlabel('Episode #')

    # Save the figure
    plt.savefig('training_metrics.png')
    plt.show()


if __name__ == '__main__':
    main()
