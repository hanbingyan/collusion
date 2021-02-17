import random
from collections import namedtuple
import numpy as np
import pickle

random.seed(12345)
np.random.seed(12345)

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
MEM_SIZE = 2250
BATCH_SIZE = 1
DELTA = 0.95

eps = 1e-5
alpha = 0.15

actions_space = np.arange(1.43, 2.0, 0.04)
quality = np.ones(2) * 2
margin_cost = np.ones(2)
horizon = 1 / 4
a0 = 0
n_actions = actions_space.size
n_agents = 2

reward_sum = np.array([5.58232796, 5.78802889, 5.92606135, 5.99644584, 6.00067233,
                       5.94172477, 5.82402394, 5.65328833, 5.43631956, 5.18072579,
                       4.89460298, 4.58619785, 4.26357789, 3.93433261, 3.60532586])



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
        ind = np.arange(self.position - batch_size, self.position)%self.capacity
        return [self.memory[i] for i in ind]

    def __len__(self):
        return len(self.memory)

def replay_classic_reward(action):
    # Compute profits for all agents

    price = actions_space[action]
    demand = np.exp((quality - price) / horizon)
    demand = demand / (np.sum(demand) + np.exp(a0 / horizon))
    reward = np.multiply(price - margin_cost, demand)
    return reward

def replay_classic_select(agent, state, steps_done):

    sample = random.random()
    eps_threshold = np.exp(-eps * steps_done)

    if sample > eps_threshold:
        return Q[agent][state[0]][state[1]].argmax()
    else:
        return np.random.randint(0, n_actions, 1, dtype=int)


def replay_classic_opt(agent, memory, BS):
    if len(memory) < BS:
        return
    transitions = memory.sample(BS)
    batch = Transition(*zip(*transitions))

    for i in range(BS):
        stat = batch.state[i]
        next_stat = batch.next_state[i]
        act = batch.action[i]
        rew = batch.reward[i]
        Q[agent][stat[0]][stat[1]][act[agent]] = (1 - alpha) * Q[agent][stat[0]][stat[1]][act[agent]] + \
                                              alpha * (rew[agent] + DELTA * Q[agent][next_stat[0]][next_stat[1]].max())
    return

Q_hist = []
end_price = []
for sess in range(500):
    steps_done = 0

    state_hist = []
    # Initialize the environment and state
    state = np.random.randint(0, n_actions, size=n_agents)
    state_hist.append(state)
    # Counter for variations in heat
    count = 0

    Q = np.zeros((n_agents, n_actions, n_actions, n_actions))
    for agent in range(n_agents):
        for i in range(n_actions):
            for j in range(n_actions):
                Q[agent, i, j, :] = reward_sum
    memory = ReplayMemory(MEM_SIZE)

    for i_episode in range(10000000):
        # For each agent, select and perform an action
        action = np.zeros(n_agents, dtype=int)

        # if i_episode == num_episodes - 100:
        #     action[0, 0] = 4
        #     action[0, 1] = select_action_classic(Q[1], state, steps_done)
        # else:
        for i in range(n_agents):
            action[i] = replay_classic_select(i, state, steps_done)

        steps_done += 1

        reward = replay_classic_reward(action)

        # Observe new state
        next_state = action

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        old_heat0 = Q[0].argmax(2)
        old_heat1 = Q[1].argmax(2)

        replay_classic_opt(0, memory, BATCH_SIZE)
        replay_classic_opt(1, memory, BATCH_SIZE)

        new_heat0 = Q[0].argmax(2)

        new_heat1 = Q[1].argmax(2)

        if np.sum(np.abs(old_heat0 - new_heat0)) == 0 and np.sum(np.abs(old_heat1 - new_heat1)) == 0:
            count += 1
        else:
            count = 0


        if i_episode%100000 == 0:
            print('Session price', sess, actions_space[action])
            print('count', count)
            print('steps done:', steps_done)

        state = next_state
        state_hist.append(state)

        if count == 100000:
            print('Terminate condition satisfied with price', np.array(state_hist[-20:]))
            break
    end_price.append(state_hist[-20:])
    Q_hist.append(Q)

with open('classic2250_rand_endprice.pickle', 'wb') as fp:
    pickle.dump(end_price, fp)

with open('classic2250_rand_Q.pickle', 'wb') as fp:
    pickle.dump(Q_hist, fp)
