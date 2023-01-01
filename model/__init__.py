import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.input_layer = nn.Linear(n_observations, 512)
        self.hidden_layer = nn.Linear(512, 512)
        self.output_layer = nn.Linear(512, n_actions)

    def forward(self, x):
        x = F.leaky_relu(self.input_layer(x), 0.1)
        x = F.leaky_relu(self.hidden_layer(x), 0.1)
        return self.output_layer(x)