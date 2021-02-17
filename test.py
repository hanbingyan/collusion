import torch
import torch.nn as nn
from models import *
from config import *
from utils import *
import numpy as np
from sklearn import datasets

import copy

# X_numpy, y_numpy = datasets.make_regression(n_samples=10, n_features=10, noise=20, random_state=1)
#
# X = torch.from_numpy(X_numpy.astype(np.float32))
# y = torch.from_numpy(y_numpy.astype(np.float32))
# y = y.view(-1, 1)
# n_samples, n_features = X.size()
#
# a = torch.ones(1)
# class Regression(nn.Module):
#
#     def __init__(self):
#         super(Regression, self).__init__()
#         self.linear = nn.Linear(10, 1)
#         nn.init.zeros_(self.linear.weight)
#         self.linear.bias = nn.Parameter(copy.deepcopy(a))
#
#     def forward(self, x):
#         return self.linear(x)
#
# net1 = Regression()
# net2 = Regression()

# print(net2(torch.ones(10)))

# class DQNm(nn.Module):
#     # Input shape: batch * recent_events_length * no_agents
#     # Outputs: batch * no_actions
#
#     # def __init__(self, recent_k, n_agents, n_actions):
#     #     super(DQN, self).__init__()
#     #     hidden = 256
#     #     self.linear1 = nn.Linear(recent_k, hidden)
#     #     # self.drop = nn.Dropout(0.2)
#     #     self.linear2 = nn.Linear(hidden, 1)
#     #     self.linear3 = nn.Linear(n_agents, n_actions)
#
#     # def forward(self, x):
#     #     x = F.relu(self.linear1(x.transpose(-1, -2)))
#     #     x = F.relu(self.linear2(x))
#     #     x = x.squeeze(-1)
#     #     x = F.relu(self.linear3(x))
#     #     return x
#
#     def __init__(self, recent_k, n_agents, n_actions, initial_Q):
#         super(DQNm, self).__init__()
#         self.linear1 = nn.Linear(recent_k * n_agents, 512)
#         # torch.nn.init.zeros_(self.linear1.weight)
#         # torch.nn.init.zeros_(self.linear1.bias)
#
#         # self.linear2 = nn.Linear(512, 512)
#
#         # torch.nn.init.zeros_(self.linear2.weight)
#         # torch.nn.init.zeros_(self.linear2.bias)
#
#         self.linear3 = nn.Linear(512, n_actions)
#         # torch.nn.init.zeros_(self.linear3.weight)
#         # torch.nn.init.zeros_(self.linear3.bias)
#         self.linear3.bias = torch.nn.Parameter(copy.deepcopy(initial_Q))
#
#     def forward(self, x):
#         x = x.view(-1, recent_k * n_agents)
#         x = F.relu(self.linear1(x))
#         # x = F.relu(self.linear2(x))
#         x = self.linear3(x)
#         return x
#
# Q = torch.randint(15, (15, 15, 15), dtype=torch.float)
#
#
# # initial = torch.zeros(15, device=device)
# # initial = AER_initial_Q()
# initial = 2*torch.rand(15) - 1
# policy_net = DQNm(1, 2, 15, initial)
# criterion = nn.MSELoss()
# optimizer = torch.optim.RMSprop(policy_net.parameters())
#
# for epoch in range(3000):
#     Q_pred = policy_net(test_batch).view(15, 15, 15)
#
#     loss = criterion(Q_pred, Q)
#
#     optimizer.zero_grad()
#
#     loss.backward()
#
#     optimizer.step()
#
#
# # print(torch.sum(torch.abs(Q_pred - Q)))
# print(policy_net(test_batch).max(1)[1].view(15, 15))


mem = ReplayMemory(10)

for i in range(10):
    mem.push(i, 2*i, 3*i, 4*i)
    print(len(mem))
for i in range(10, 20):
    print(len(mem))
    mem.push(i, 2*i, 3*i, 4*i)
    print(i, mem.recent(10))