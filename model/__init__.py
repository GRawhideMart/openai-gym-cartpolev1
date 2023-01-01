import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.input_layer = nn.Linear(n_observations, 256)
        self.hidden_layer = nn.Linear(256, 256)
        self.output_layer = nn.Linear(256, n_actions)

    def forward(self, x):
        x = F.relu(self.input_layer)
        x = F.relu(x)
        return self.output_layer(x)