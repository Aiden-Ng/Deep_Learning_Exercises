
"""
Version: 1.2

1. Implementing Backward propogation and gradient descent

--Minor changes by me

--Minor changes by me
"""


import gymnasium
import flappy_bird_gymnasium
import torch
from torch import nn
from experience_replay import ReplayMemory
import itertools
import yaml
import random



from dqn import DQN

#check if gpu calculation is supported for tensorflow
device = "cuda" if torch.cuda.is_available() else "cpu"
# if u wanna force use cpu
#device = "cpu"

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
        self.network_sync_rate = hyperparameters["network_sync_rate"]
        self.learning_rate_a = hyperparameters["learning_rate_a"]
        self.discount_factor_g = hyperparameters["discount_factor_g"]

        self.loss_fn = nn.MSELoss()
        self.optimizer = None
        
    
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

            target_dqn = DQN(num_states, num_actions).to(device)
            # sync the target network to the policy network
            # copies all the weights and biases into the target network
            target_dqn.load_state_dict(policy_dqn.state_dict())

            # count how many steps that have been taken
            step_count = 0

            # policy network optimizer. "Adam" optimizer can be swapped to something else
            # q: what does this network optimizer does
            # the policy_dqn.parameters() is the parameters that we want to optimize
            self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr = self.learning_rate_a)
        
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

                    # increment step count
                    step_count += 1
                
                # move to new state
                state = new_state
            
            rewards_per_episode.append(episode_reward)
            print(episode_reward)

            epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
            epsilon_history.append(epsilon)

            #if enough the experience have been loaded
            if len(memory) > self.mini_batch_size:
                # sample form memory
                mini_batch = memory.sample(self.mini_batch_size)

                # q: do not understand hat does this mean
                self.optimize(mini_batch, policy_dqn, target_dqn)

                # copy policy network to target network after a certain number of steps
                if step_count > self.network_sync_rate:
                    # copy the policy network to the target network
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                    step_count = 0

                 
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
    agent = Agent("cartpole1")
    agent.run(is_training = True, render = True)

