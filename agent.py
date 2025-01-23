
"""
Version: 1.0

1. This is a single DQN agent 
2. Run and managed to compile the code
"""

import flappy_bird_gymnasium
import gymnasium
import torch
from experience_replay import ReplayMemory
import itertools
import yaml
import random


from dqn import DQN

#check if gpu calculation is supported for tensorflow
device = "cuda" if torch.cuda.is_available() else "cpu"

class Agent:
    def __init__(self, hyperparameters_sets):
        with open("hyperparameters.yml", "r") as file:
            all_hyperparameters_sets = yaml.safe_load(file)
            hyperparameters = all_hyperparameters_sets[hyperparameters_sets]
          
        self.replay_memory_size = hyperparameters["replay_memory_size"]
        self.mini_batch_size = hyperparameters["mini_batch_size"]
        self.epsilon_init = hyperparameters["epsilon_init"]
        self.epsilon_decay = hyperparameters["epsilon_decay"]
        self.epsilon_min = hyperparameters["epsilon_min"]
        
    
    def run(self, is_training = True, render = False):
        #env = gymnasium.make("FlappyBird-v0", render_mode="human", use_lidar=False)
        env = gymnasium.make("CartPole-v1", render_mode="human" if render else None)
         
        #number of states
        num_states = env.observation_space.shape[0]
        
        #number of action
        num_actions = env.action_space.n

        # list to contain the rewards per episode
        rewards_per_episode = []

        # list to store the epsilon histories
        epsilon_history = []

        #q:  what does this does?
        #a:  it initializes the policy network
        #q: what does the to_device do?
        #a: it moves the model to the device
        policy_dqn = DQN(num_states, num_actions).to(device)

        if is_training:
            memory = ReplayMemory(self.replay_memory_size)

            epsilon = self.epsilon_init
        
        for episode in itertools.count():
            state, _ = env.reset()
            
            #it seems like tensor is a data struct that is used for gpu computations
            state = torch.tensor(state, dtype = torch.float, device = device)

            terminated = False
            episode_reward = 0.0

            while not terminated:
                # Next action:
                # (feed the observation to your agent here)
                if is_training and random.random() < epsilon:
                    #what action is this sampling
                    action = env.action_space.sample()
                    action = torch.tensor(action, dtype = torch.int64, device = device)
                else:
                    # to not do calculation during training, we can turn it off my using torch.no_grad()
                    with torch.no_grad():
                        # since pytorch uses the first batch dimension as the batch size
                        # we need to add a batch dimension to the state
                        # from tensor([1,2,3]) ===> tensor([[1,2,3]])
                        action = policy_dqn(state.unsqueeze(dim = 0)).squeeze().argmax()

                # Processing:
                new_state, reward, terminated, _, info = env.step(action.item())

                # accumulate the reward
                episode_reward += reward

                new_state = torch.tensor(new_state, dtype = torch.float, device = device)

                if is_training:
                    memory.append((state, action, new_state, reward, terminated))
                
                # move to new state
                state = new_state
            
            rewards_per_episode.append(episode_reward)
            print(episode_reward)

            epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
            epsilon_history.append(epsilon)
        
        

if __name__ == "__main__":
    agent = Agent("cartpole1")
    agent.run(is_training = True, render = True)

