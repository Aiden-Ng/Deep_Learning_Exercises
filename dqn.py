import torch
from torch import nn
import torch.nn.functional as F

# q: what is a nn.Module?
# a: nn.Module is the base class for all neural network modules in PyTorch.

class DQN(nn.Module):
    #explain the contento f the __init__ function
    def __init__(self, state_dim, action_dim, hidden_dim = 256):
        #q: what does super(DQN, self).__init__() do?
        #a: super(DQN, self).__init__() calls the __init__ method of the parent class of DQN, which is nn.Module.
        super(DQN, self).__init__()

        # in pyTorch the input layer is inplicit, and you do not need to declare it
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    # forward function that does the calculation
    def forward(self, x):
        #q: what does F.relu do?
        #a: F.relu is the rectified linear unit activation function.

        #q: what does the activation function do?
        #a: The activation function is used to introduce non-linearity to the output of a neuron.
        x = F.relu(self.fc1(x))
        return self.fc2(x)
        

if __name__ == "__main__":
    #q: what does the following code do?
    #a: The following code creates an instance of the DQN class and prints the model architecture.
    
    state_dim = 12
    action_dim = 2
    net = DQN(state_dim, action_dim)   
    state = torch.randn(10, state_dim)
    output = net(state)
    print(output)
        
        
