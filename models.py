import torch.nn as nn
import torch.nn.functional as F
import copy

from config import *

class DQN(nn.Module):
    # Input shape: batch * recent_events_length * num_agents
    # Outputs: batch * num_actions

    def __init__(self, recent_k, n_agents, n_actions, initial_Q):
        super(DQN, self).__init__()
        self.linear1 = nn.Linear(recent_k * n_agents, 512)
        torch.nn.init.zeros_(self.linear1.weight)
        torch.nn.init.zeros_(self.linear1.bias)

        self.linear2 = nn.Linear(512, n_actions)
        torch.nn.init.zeros_(self.linear2.weight)
        # torch.nn.init.zeros_(self.linear2.bias)
        self.linear2.bias = torch.nn.Parameter(copy.deepcopy(initial_Q))

    def forward(self, x):
        x = x.view(-1, recent_k * n_agents)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x
