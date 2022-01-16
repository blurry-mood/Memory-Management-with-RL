from environment import MemoryEnvironment
from os.path import split, join
from config import *
import matplotlib.pyplot as plt
from tqdm import tqdm
from random import choice

_CURRENT_DIR = split(__file__)[0]


env = MemoryEnvironment(n_slots=N_SLOTS, n_arrays=N_ARRAYS, max_size=MAX_ARRAY_SIZE)

actions_map = [(i, j, k) for i, j, k in zip(range(N_SLOTS + N_ARRAYS), range(N_SLOTS + N_ARRAYS), range(2))]

rewards = []
for i in range(EPISODES):
    state = env.reset()
    _reward = 0
    n = 0
    done = False
    with tqdm(desc=f'Episode {i+1}') as pbar:
        while not done:
            n += 1

            action = choice(actions_map)
            state, reward, done, info = env.step(action)

            _reward += reward

            pbar.update(1)
            pbar.set_postfix({'Cumulative reward': _reward})

    rewards.append(_reward)

env.close()

plt.plot(rewards)
plt.xlabel('Episodes')
plt.ylabel('Cumulative Reward')
plt.title('Evolution of Cumulative Reward')
plt.savefig('random_agent')
plt.show()