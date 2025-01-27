
"""
Version: 1.4
Video number 9
1. Test the DQN algorithm on flappybird_v0

--Minor changes by me

--Minor changes by me

Notes: 
1. Recheck the epsilon array
"""

import gymnasium
import numpy as np

import matplotlib 
import matplotlib.pyplot as plt

import random
import torch
from torch import nn
import yaml

from experience_replay import ReplayMemory
from dqn import DQN

from datetime import datetime, timedelta
import argparse
import itertools

import flappy_bird_gymnasium
import os

# for printing date and time 
DATE_FORMAT = "%m-%d %H:%M:%S"

#Directory for saving run info
curr_dir = os.path.abspath(os.path.dirname(__file__))
RUNS_DIR = "run"
os.makedirs(f"{curr_dir}/{RUNS_DIR}", exist_ok = True)

# "Agg" : used to generate plot as images nad save them to a file instead of rendering to screen
# Agg is a non-interactive back-end
matplotlib.use("Agg")

#check if gpu calculation is supported for tensorflow
device = "cuda" if torch.cuda.is_available() else "cpu"
# if u wanna force use cpu
device = "cpu"

class Agent:
    def __init__(self, hyperparameters_sets):
        with open("hyperparameters.yml", "r") as file:
            all_hyperparameters_sets = yaml.safe_load(file)
            hyperparameters = all_hyperparameters_sets[hyperparameters_sets]
          
        self.hyperparameters_sets = hyperparameters_sets

        # Hyperparameters (adjustable)
        self.env_id             = hyperparameters['env_id']
        self.learning_rate_a    = hyperparameters['learning_rate_a']        # learning rate (alpha)
        self.discount_factor_g  = hyperparameters['discount_factor_g']      # discount rate (gamma)
        self.network_sync_rate  = hyperparameters['network_sync_rate']      # number of steps the agent takes before syncing the policy and target network
        self.replay_memory_size = hyperparameters['replay_memory_size']     # size of replay memory
        self.mini_batch_size    = hyperparameters['mini_batch_size']        # size of the training data set sampled from the replay memory
        self.epsilon_init       = hyperparameters['epsilon_init']           # 1 = 100% random actions
        self.epsilon_decay      = hyperparameters['epsilon_decay']          # epsilon decay rate
        self.epsilon_min        = hyperparameters['epsilon_min']            # minimum epsilon value
        self.stop_on_reward     = hyperparameters['stop_on_reward']         # stop training after reaching this number of rewards
        self.fc1_nodes          = hyperparameters['fc1_nodes']
        self.env_make_params    = hyperparameters.get('env_make_params',{}) # Get optional environment-specific parameters, default to empty dict

        # the right side is the deafult value if it is not found in the dictionary
        self.use_lidar = hyperparameters['env_make_params'].get('use_lidar', None)
        

        self.loss_fn = nn.MSELoss()
        self.optimizer = None

        # Path to run info
        # what does os.path.join does
        # it joins the path with the file name
        self.LOG_FILE = os.path.join(f"{curr_dir}/{RUNS_DIR}", f"{self.hyperparameters_sets}.log")
        self.MODEL_FILE = os.path.join(f"{curr_dir}/{RUNS_DIR}", f"{self.hyperparameters_sets}.pt") # this is a pytorch file
        self.GRAPH_FILE = os.path.join(f"{curr_dir}/{RUNS_DIR}", f"{self.hyperparameters_sets}.png")
        
    
    def run(self, is_training = True, render = False):
        if is_training:
            start_time = datetime.now()
            last_graph_update_time = start_time

            log_message = f"{start_time.strftime(DATE_FORMAT)}: Training starting..."
            print(log_message)
            with open(self.LOG_FILE, 'w') as file:
                file.write(log_message + '\n')
        
        #env = gymnasium.make("FlappyBird-v0", render_mode="human", use_lidar=False)
        env = gymnasium.make(self.env_id , render_mode="human" if render else None, use_lidar = self.use_lidar)
        
         
        #number of states
        num_states = env.observation_space.shape[0]
        
        #number of action
        num_actions = env.action_space.n

        # list to contain the rewards per episode
        rewards_per_episode = []

        #q:  what does this does?
        #a:  it initializes the policy network
        #q: what does the to_device do?
        #a: it moves the model to the device
        policy_dqn = DQN(num_states, num_actions).to(device)

        if is_training:
            memory = ReplayMemory(self.replay_memory_size)

            epsilon = self.epsilon_init

            target_dqn = DQN(num_states, num_actions).to(device)
            # sync the target network to the policy network
            # copies all the weights and biases into the target network
            target_dqn.load_state_dict(policy_dqn.state_dict())

            # list to store the epsilon histories
            epsilon_history = []
            
            # count how many steps that have been taken
            step_count = 0

            # policy network optimizer. "Adam" optimizer can be swapped to something else
            # q: what does this network optimizer does
            # the policy_dqn.parameters() is the parameters that we want to optimize
            self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr = self.learning_rate_a)

            # track best rewards
            best_reward = -999999
        
        else:
            # load learned policy
            policy_dqn.load_state_dict(torch.load(self.MODEL_FILE))

            # switch model to evaluation mode
            policy_dqn.eval()

        for episode in itertools.count():
            
            state, _ = env.reset()
            
            #it seems like tensor is a data struct that is used for gpu computations
            state = torch.tensor(state, dtype = torch.float, device = device)

            terminated = False
            episode_reward = 0.0

            while (not terminated and episode_reward < self.stop_on_reward):
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
                # Execute action. Truncated and info not used
                new_state, reward, terminated, truncated, info = env.step(action.item())

                # accumulate the reward
                episode_reward += reward

                new_state = torch.tensor(new_state, dtype = torch.float, device = device)

                if is_training:
                    memory.append((state, action, new_state, reward, terminated))

                    # increment step count
                    step_count += 1
                
                # move to new state
                state = new_state
            
            rewards_per_episode.append(episode_reward)

            # Save the model when new best reward is obtained
            if is_training:
                if episode_reward > best_reward:
                    log_message = f"{datetime.now().strftime(DATE_FORMAT)}: New best reward {episode_reward:0.1f} ({(episode_reward-best_reward)/best_reward*100:+.1f}%) at episode {episode}, saving model..."
                    print(log_message)
                    with open(self.LOG_FILE, 'a') as file:
                        file.write(log_message + '\n')

                    torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
                    best_reward = episode_reward
            
                # Update graph every x seconds
                current_time = datetime.now()
                if current_time - last_graph_update_time > timedelta(seconds=10):
                    self.save_graph(rewards_per_episode, epsilon_history)
                    last_graph_update_time = current_time
            
                #if enough the experience have been loaded
                if len(memory) > self.mini_batch_size:
                    # sample form memory
                    mini_batch = memory.sample(self.mini_batch_size)

                    # q: do not understand hat does this mean
                    self.optimize(mini_batch, policy_dqn, target_dqn)

                    # Decay epsilon QUESTION, might not work
                    epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
                    epsilon_history.append(epsilon)


                    # copy policy network to target network after a certain number of steps
                    if step_count > self.network_sync_rate:
                        # copy the policy network to the target network
                        target_dqn.load_state_dict(policy_dqn.state_dict())
                        step_count = 0


    def save_graph(self, rewards_per_episode, epsilon_history):
        # Save plots
        fig = plt.figure(1)

        # Plot average rewards (Y-axis) vs episodes (X-axis)
        mean_rewards = np.zeros(len(rewards_per_episode))
        for x in range(len(mean_rewards)):
            mean_rewards[x] = np.mean(rewards_per_episode[max(0, x-99):(x+1)])
        plt.subplot(121) # plot on a 1 row x 2 col grid, at cell 1
        # plt.xlabel('Episodes')
        plt.ylabel('Mean Rewards')
        plt.plot(mean_rewards)

        # Plot epsilon decay (Y-axis) vs episodes (X-axis)
        plt.subplot(122) # plot on a 1 row x 2 col grid, at cell 2
        # plt.xlabel('Time Steps')
        plt.ylabel('Epsilon Decay')
        plt.plot(epsilon_history)

        plt.subplots_adjust(wspace=1.0, hspace=1.0)

        # Save plots
        fig.savefig(self.GRAPH_FILE)
        plt.close(fig)


    # updating the q value
    def optimize(self, mini_batch, policy_dqn, target_dqn):
        
        # Transpose the list of experiences and separate each element

        states, actions, new_states, rewards, terminations = zip(*mini_batch)

        # Stack tensors to create batch tensors

        states = torch.stack(states)

        actions = torch.stack(actions)

        new_states = torch.stack(new_states)
        
        
        # edited by me, to convert the tuple of floats to a tuple of tensor
        # this is to ensure that rewards = torch,stack(rewards works)
        rewards = tuple(torch.tensor(reward, dtype = torch.float, device = device) for reward in rewards)
        rewards = torch.stack(rewards)

        terminations = torch.tensor(terminations).float().to(device)

        with torch.no_grad():
            # Compute the target q value
            # if terminated, then target_q = rewards
            # else target_q = rewards +  discount_factor_g * target_dqn(new_states).max(dim = 1).values
            target_q = rewards + ( 1- terminations ) * self.discount_factor_g * target_dqn(new_states).max(dim = 1).values
            '''
                target_dqn(new_states)  ==> tensor([[1,2,3],[4,5,6]])
                    .max(dim=1)         ==> torch.return_types.max(values=tensor([3,6]), indices=tensor([3, 0, 0, 1]))
                        [0]             ==> tensor([3,6])
            '''
        # Calcuate Q values from current policy
        current_q = policy_dqn(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()
        '''
            policy_dqn(states)  ==> tensor([[1,2,3],[4,5,6]])
                actions.unsqueeze(dim=1)
                .gather(1, actions.unsqueeze(dim=1))  ==>
                    .squeeze()                    ==>
        '''
                
        #compute loss for the whole minibatch
        loss = self.loss_fn(current_q, target_q) 

        #Optimize the model
        self.optimizer.zero_grad() #clear gradient
        loss.backward()            #compute gradient (backpropagation)
        self.optimizer.step()      #Update network parameters i.e. weights and biases

        

if __name__ == "__main__":
    # Parse command line inputs
    parser = argparse.ArgumentParser(description='Train or Test models')
    parser.add_argument("hyperparameters", help = "")
    parser.add_argument("--train", help = "Training mode", action="store_true")
    args = parser.parse_args()

    dql = Agent(hyperparameters_sets = args.hyperparameters)

    if args.train:
        dql.run(is_training= True, render = False)
    else: 
        dql.run(is_training= False, render = True)


