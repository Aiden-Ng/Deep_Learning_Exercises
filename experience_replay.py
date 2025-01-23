from collections import deque
import random

class ReplayMemory():
    def __init__(self, maxlen, seed = None):
        self.memory = deque([], maxlen = maxlen)

        #Optional seed for reproducability
        if seed is not None:
            random.seed(seed)

    #add one transition to the experience replay
    # q: what is a transition?
    # a: A transition is a tuple of (state, action, reward, next_state, done).
    def append(self, transition):
        self.memory.append(transition)
    
    # what is a batch size
    # a: A batch size is the number of transitions to sample from the experience replay buffer.
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)
