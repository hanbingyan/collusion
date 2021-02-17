import torch.optim as optim
import copy

import pickle

from utils import *
from models import DQN

initial_Q = AER_initial_Q()
# initial_Q = torch.zeros(n_actions, device=device)
# initial_Q[-1] = n_actions - 1
policy_net = []
target_net = []
optimizer = []

for i in range(n_agents):
    policy_net.append(DQN(recent_k, n_agents, n_actions, initial_Q).to(device))
    target_net.append(DQN(recent_k, n_agents, n_actions, initial_Q).to(device))
    target_net[i].load_state_dict(policy_net[i].state_dict())
    target_net[i].eval()
    optimizer.append(optim.RMSprop(policy_net[i].parameters()))

memory = ReplayMemory(MEM_SIZE)

# random_hist0 = []
# recent_hist0 = []
# random_hist1 = []
# recent_hist1 = []

heat = torch.zeros(n_agents, n_actions, n_actions, device=device)
heat_episode = torch.zeros(num_sub, n_agents, n_actions, n_actions, device=device)
heat_unique0 = []
heat_freq0 = []
heat_unique1 = []
heat_freq1 = []


test_action = torch.zeros(n_agents, n_actions**2, device=device)

steps_done = torch.zeros(1, device=device)


for i_episode in range(num_episodes):
    # Initialize the environment and state
    heat_episode = torch.zeros(num_sub, n_agents, n_actions, n_actions, device=device)

    state = torch.randint(0, n_actions, size=(recent_k, n_agents), dtype=torch.float,
                          device=device)
    for t in range(num_sub):
        # For each agent, select and perform an action
        action = torch.zeros(1, n_agents, device=device)
        # Force defection
        # if i_episode == num_episodes - 1 and t == 1:
        #     action[0, 0] = 14
        #     action[0, 1] = select_action_deep(policy_net[1], state, memory, steps_done)
        # else:
        for i in range(n_agents):
            action[0, i] = select_action_deep(policy_net[i], state, memory, steps_done)
        if i_episode == num_episodes - 1:
            print('State in last episode', t, state)
            print('Action in last episode', t, action)
        steps_done += 1
        reward = reward_comp(action.view(-1))

        # Observe new state
        next_state = torch.cat((state[1:], action.view(1, -1)), dim=0)

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        state = copy.deepcopy(next_state)

        # Test params difference after optim
        # befopt_dict0 = copy.deepcopy(policy_net[0].state_dict())
        # befopt_dict1 = copy.deepcopy(policy_net[1].state_dict())
        for i in range(n_agents):
            # if len(memory) < BATCH_SIZE:
            # Perform one step of the optimization (on the target network)
            optimize_model(i, policy_net[i], target_net[i], memory, optimizer[i])
            # else:
            #     random_corr, recent_corr = optimize_model(i, policy_net[i], target_net[i], memory, optimizer[i])
            #     if i == 0:
            #         random_hist0.append(random_corr)
            #         recent_hist0.append(recent_corr)
            #     else:
            #         random_hist1.append(random_corr)
            #         recent_hist1.append(recent_corr)

        # aftopt_dict0 = copy.deepcopy(policy_net[0].state_dict())
        # aftopt_dict1 = copy.deepcopy(policy_net[1].state_dict())

        # print('Parameter diff', torch.sum(torch.abs(aftopt_dict0['linear3.bias'] - befopt_dict0['linear3.bias'])))

        for k in range(n_agents):
            test_action[k, :] = policy_net[k](test_batch).max(1)[1].detach()
            heat[k, :, :] = test_action[k, :].view(n_actions, n_actions)

        heat_episode[t, :, :, :] = heat

    print('Action before optimization', action)

    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        for i in range(n_agents):
            target_net[i].load_state_dict(policy_net[i].state_dict())

    print('price', actions_space[action.view(-1).long()].to('cpu').numpy())
    # print('reward', reward.to('cpu').numpy())
    print('steps done:', steps_done.item())

    uniq0, freq0 = torch.unique(heat_episode[:, 0, :, :], sorted = True, return_counts=True)
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
# Use name Donline instead, for online algo
with open('Drand_heat_unique0.pickle', 'wb') as fp:
    pickle.dump(heat_unique0, fp)

with open('Drand_heat_unique1.pickle', 'wb') as fp:
    pickle.dump(heat_unique1, fp)

with open('Drand_heat_freq0.pickle', 'wb') as fp:
    pickle.dump(heat_freq0, fp)

with open('Drand_heat_freq1.pickle', 'wb') as fp:
    pickle.dump(heat_freq1, fp)

# Save last episode all heat values
with open('Drand_lastepisode_heat.pickle', 'wb') as fp:
    pickle.dump(heat_episode.to('cpu').numpy(), fp)

# Save the last network weights
torch.save(policy_net[0].state_dict(), 'Drand_policy_net0.pth')
torch.save(policy_net[1].state_dict(), 'Drand_policy_net1.pth')

# Save correlations
# with open('deep_random_corr0.pickle', 'wb') as fp:
#     pickle.dump(random_hist0, fp)
#
# with open('deep_random_corr1.pickle', 'wb') as fp:
#     pickle.dump(random_hist1, fp)
#
# with open('deep_recent_corr0.pickle', 'wb') as fp:
#     pickle.dump(recent_hist0, fp)
#
# with open('deep_recent_corr1.pickle', 'wb') as fp:
#     pickle.dump(recent_hist1, fp)

# random_hist0 = np.array(random_hist0)
# random_hist1 = np.array(random_hist1)
# recent_hist0 = np.array(recent_hist0)
# recent_hist1 = np.array(recent_hist1)
# np.savetxt("deep_random_corr0.csv", random_hist0, delimiter=",")
# np.savetxt("deep_recent_corr0.csv", recent_hist0, delimiter=",")
# np.savetxt("deep_random_corr1.csv", random_hist1, delimiter=",")
# np.savetxt("deep_recent_corr1.csv", recent_hist1, delimiter=",")

# Save two figures of the last heat
# plt.figure(figsize=(8, 6))
# ax = sns.heatmap(heat[0].to('cpu').numpy(), cbar=False, annot=True)
# fig = ax.get_figure()
# fig.savefig('deep_heat0.eps', format='eps', dpi=500, bbox_inches='tight', pad_inches=0.1)
#
# plt.figure(figsize=(8, 6))
# ax1 = sns.heatmap(heat[1].to('cpu').numpy(), cbar=False, annot=True)
# fig1 = ax1.get_figure()
# fig1.savefig('deep_heat1.eps', format='eps', dpi=500, bbox_inches='tight', pad_inches=0.1)

print('heat0', heat[0])
print('heat1', heat[1])
