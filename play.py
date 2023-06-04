from unityagents import UnityEnvironment
import numpy as np
from agent import *

def main():

    # Load the environment
    env = UnityEnvironment(file_name="Tennis_Windows_x86_64/Tennis.exe")

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment
    env_info = env.reset(train_mode=False)[brain_name]

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

    # Load the agent
    multi_agent = MADDPG(num_agents, state_size, action_size, RANDOM_SEED)
    
    # Load the weights from file
    multi_agent.load('tennis_actor', 'tennis_critic')

    while True:
        actions = multi_agent.act(states)                       # select an action (for each agent)
        env_info = env.step(actions)[brain_name]           # send all actions to the environment
        next_states = env_info.vector_observations         # get next state (for each agent)
        rewards = env_info.rewards                         # get reward (for each agent)
        dones = env_info.local_done                        # see if episode finished
        states = next_states                               # roll over states to next time step
        if np.any(dones):                                  # exit loop if episode finished
            break

    env.close()

if __name__ == "__main__":
    main()
