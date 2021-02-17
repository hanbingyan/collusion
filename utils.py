import random
from collections import namedtuple
import torch.nn.functional as F
from config import *

import numpy as np

random.seed(12345)
np.random.seed(12345)

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

### Code for ReplayMemory class is modified from the online tutorial at
### https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
### Author: Adam Paszke
class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def recent(self, batch_size):
        """The latest tuples"""
        ind = np.arange(self.position - batch_size, self.position) % self.capacity
        return [self.memory[i] for i in ind]

    def __len__(self):
        return len(self.memory)


def reward_comp(action):
    # Compute profits for all agents
    # Input: actions taken by all agents, shape: n_agents;
    # Output: profits for all agents
    action = action.long()
    price = actions_space[action]
    demand = torch.exp((quality - price) / horizon)
    demand = demand / (torch.sum(demand) + torch.exp(a0 / horizon))
    reward = torch.mul(price - margin_cost, demand)
    return reward.view(1, -1)

# Only works for num_agents = 2
def AER_initial_Q():
    reward_sum = torch.zeros(n_actions, device=device, requires_grad=False)
    for self_act in range(n_actions):
        for rival_act in range(n_actions):
            if rival_act == self_act:
                continue
            state = torch.tensor([self_act, rival_act], device=device)
            reward_sum[self_act] += reward_comp(state)[0, 0]
    reward_sum = reward_sum/(1 - DELTA)/n_actions
    return reward_sum


def select_action_classic(Q, state, steps_done):
    sample = random.random()
    eps_threshold = torch.exp(-eps * steps_done)

    state_idx = state.view(-1).int()
    if sample > eps_threshold:
        return Q[state_idx[0], state_idx[1]].argmax().float()
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.float)


def select_action_deep(policy_net, state, memory, steps_done):
    sample = random.random()
    eps_threshold = torch.exp(-eps * steps_done)

    if len(memory) >= BATCH_SIZE and sample > eps_threshold:
        with torch.no_grad():
            batch_state = state.unsqueeze(0)
            return policy_net(batch_state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.float)


def optimize_model(agent, policy_net, target_net, memory, optimizer):
    if len(memory) < BATCH_SIZE:
        return

    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))
    next_states = torch.stack(batch.next_state)
    state_batch = torch.stack(batch.state)
    action_batch = torch.cat(batch.action).long()
    reward_batch = torch.cat(batch.reward)

    # recent_trans = memory.recent(BATCH_SIZE)
    # recent_batch = Transition(*zip(*recent_trans))
    # recent_states = torch.stack(recent_batch.state)
    #
    # random_corr = np.corrcoef(state_batch.to('cpu').numpy()[:-1, 0, agent], state_batch.to('cpu').numpy()[1:, 0, agent])[1, 0]
    # recent_corr = np.corrcoef(recent_states.to('cpu').numpy()[:-1, 0, agent], recent_states.to('cpu').numpy()[1:, 0, agent])[1, 0]
    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net

    state_action_values = policy_net(state_batch).gather(1, action_batch[:, agent].view(-1, 1))
    # state_action_values = state_action_values[:, agent].view(-1, 1)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    next_state_values = target_net(next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * DELTA) + reward_batch[:, agent]

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

    return
    # return random_corr, recent_corr
