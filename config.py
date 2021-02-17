import torch
# check gpu is available

torch.manual_seed(12345)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MEM_SIZE = 2000
BATCH_SIZE = 128
DELTA = 0.95
TARGET_UPDATE = 10

eps = 1e-5
alpha = 0.15

actions_space = torch.arange(1.43, 2.0, 0.04, device=device)
# actions_space = torch.arange(1.15, 2.4, 0.05, device=device)
n_actions = actions_space.size(0)
n_agents = 2
recent_k = 1

quality = torch.ones(n_agents, device=device) * 2
margin_cost = torch.ones(n_agents, device=device)
# margin_cost[1] = 0.5
horizon = torch.ones(1, device=device) / 4
a0 = 0

num_episodes = 2000
num_sub = 500

# Batch used for heatmap computing
test_batch = torch.zeros(n_actions**2, recent_k, n_agents, device=device)
for i in range(n_actions**2):
    test_batch[i, 0, 0] = i//n_actions
    test_batch[i, 0, 1] = i%n_actions

