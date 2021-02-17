import torch.optim as optim

import copy
import pickle

from utils import *
from models import DQN

initial_Q = AER_initial_Q()
# initial_Q = torch.zeros(n_actions, device=device)


policy_net = DQN(recent_k, n_agents, n_actions, initial_Q).to(device)
target_net = DQN(recent_k, n_agents, n_actions, initial_Q).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
optimizer = optim.RMSprop(policy_net.parameters())

Q = torch.zeros(n_actions, n_actions, n_actions, device=device)
for i in range(n_actions):
    for j in range(n_actions):
        Q[i, j, :] = initial_Q.view(-1)

memory = ReplayMemory(MEM_SIZE)


heat = torch.zeros(n_agents, n_actions, n_actions, device=device)

heat_unique0 = []
heat_freq0 = []
heat_unique1 = []
heat_freq1 = []

test_action = torch.zeros(n_agents, n_actions**2, device=device)

steps_done = torch.zeros(1, device=device)

l_state = torch.ones(size=(recent_k, n_agents), dtype=torch.float, device=device)*(-1)

for i_episode in range(num_episodes):
    # Initialize the environment and state
    heat_episode = torch.zeros(num_sub, n_agents, n_actions, n_actions, device=device)

    state = torch.randint(0, n_actions, size=(recent_k, n_agents), dtype=torch.float,
                          device=device)
    for t in range(num_sub):
        # For each agent, select and perform an action
        action = torch.zeros(1, n_agents, device=device)

        # Step in and force deep player to deviate
        # if i_episode == num_episodes - 1 and t == 1:
        #     action[0, 0] = 1
        #     action[0, 1] = select_action_classic(Q, state, steps_done)
        # else:
        action[0, 0] = select_action_deep(policy_net, state, memory, steps_done)
        action[0, 1] = select_action_classic(Q, state, steps_done)
        if i_episode == num_episodes - 1:
            print('State in last episode', t, state)
            print('Action in last episode', t, action)
        steps_done += 1
        reward = reward_comp(action.view(-1))

        # Observe new state
        next_state = torch.cat((state[1:], action.view(1, -1)), dim=0)

        # Store the transition in memory
        if reward[0, 0] <= reward[0, 1]:
            memory.push(state, action, next_state, reward)

        # Perform one step of the optimization (on the target network)
        optimize_model(0, policy_net, target_net, memory, optimizer)

        # Store the most recent loss cell for classic player
        if reward[0, 0] >= reward[0, 1]:
            l_state = state
            l_reward = reward
            l_next_state = next_state
            l_action = action
        # Check if loss cell is non-empty
        if l_state[0, 0] >= 0:
            state_idx = l_state.view(-1).int()
            next_state_idx = l_next_state.view(-1).int()
            action_idx = l_action.view(-1).int()
            Q[state_idx[0], state_idx[1], action_idx[1]] = (1 - alpha) * Q[state_idx[0], state_idx[1], action_idx[1]] + \
                                                           alpha * (reward[0, 1] + DELTA * Q[next_state_idx[0],
                                                                                             next_state_idx[1]].max())
        # Move to next state
        state = copy.deepcopy(next_state)


        heat[0, :, :] = policy_net(test_batch).max(1)[1].detach().view(n_actions, n_actions)
        heat[1, :, :] = Q.argmax(2)
        heat_episode[t, :, :, :] = heat


    print('Action before optimization', action)

    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())
        # dict_hist.append(copy.deepcopy(policy_net.state_dict()))

    print('price', actions_space[action.view(-1).long()].to('cpu').numpy())
    # print('reward', reward.to('cpu').numpy())
    print('steps done:', steps_done.item())

    uniq0, freq0 = torch.unique(heat_episode[:, 0, :, :], sorted=True, return_counts=True)
    heat_unique0.append(uniq0.to('cpu').numpy())
    heat_freq0.append(freq0.to('cpu').numpy())

    uniq1, freq1 = torch.unique(heat_episode[:, 1, :, :], sorted=True, return_counts=True)
    heat_unique1.append(uniq1.to('cpu').numpy())
    heat_freq1.append(freq1.to('cpu').numpy())

    print('Agent 0 heat unique', heat_unique0[-1])
    print('Agent 0 heat freq', heat_freq0[-1])
    print('Agent 1 heat unique', heat_unique1[-1])
    print('Agent 1 heat freq', heat_freq1[-1])

# Save heat unique values, frequencies
with open('cdrk_heat_unique0.pickle', 'wb') as fp:
    pickle.dump(heat_unique0, fp)

with open('cdrk_heat_unique1.pickle', 'wb') as fp:
    pickle.dump(heat_unique1, fp)

with open('cdrk_heat_freq0.pickle', 'wb') as fp:
    pickle.dump(heat_freq0, fp)

with open('cdrk_heat_freq1.pickle', 'wb') as fp:
    pickle.dump(heat_freq1, fp)

# Save last episode all heat values
with open('cdrk_lastepisode_heat.pickle', 'wb') as fp:
    pickle.dump(heat_episode.to('cpu').numpy(), fp)

# Save the last network weights
torch.save(policy_net.state_dict(), 'cdrk_policy_net.pth')
with open('cdrk_classic_Q.pickle', 'wb') as fp:
    pickle.dump(Q.to('cpu').numpy(), fp)


# Save correlations
# with open('cdrk_random_corr0.pickle', 'wb') as fp:
#     pickle.dump(random_hist0, fp)

# with open('cdrk_random_corr1.pickle', 'wb') as fp:
#     pickle.dump(random_hist1, fp)

# with open('cdrk_recent_corr0.pickle', 'wb') as fp:
#     pickle.dump(recent_hist0, fp)

# with open('cdrk_recent_corr1.pickle', 'wb') as fp:
#     pickle.dump(recent_hist1, fp)

# # Save two figures of the last heat
# plt.figure(figsize=(8, 6))
# ax = sns.heatmap(heat[0].to('cpu').numpy(), cbar=False, annot=True)
# fig = ax.get_figure()
# fig.savefig('cdrk_heat0.eps', format='eps', dpi=500, bbox_inches='tight', pad_inches=0.1)
#
# plt.figure(figsize=(8, 6))
# ax1 = sns.heatmap(heat[1].to('cpu').numpy(), cbar=False, annot=True)
# fig1 = ax1.get_figure()
# fig1.savefig('cdrk_heat1.eps', format='eps', dpi=500, bbox_inches='tight', pad_inches=0.1)

print('heat0', heat[0])
print('heat1', heat[1])
