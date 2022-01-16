from environment import MemoryEnvironment
from os.path import split, join
from rl_algorithms import QLearning
from config import *
import matplotlib.pyplot as plt
from tqdm import tqdm

_CURRENT_DIR = split(__file__)[0]


class MemoryAgent(QLearning):
    def decode_state(self, state):
        if state is None:
            return None
        return tuple(state)


env = MemoryEnvironment(n_slots=N_SLOTS, n_arrays=N_ARRAYS, max_size=MAX_ARRAY_SIZE)

actions_map = [(i, j, k) for i, j, k in zip(range(N_SLOTS + N_ARRAYS), range(N_SLOTS + N_ARRAYS), range(2))]
qlearning = MemoryAgent(actions=list(range(len(actions_map))), alpha=ALPHA, gamma=GAMMA, eps=EPS)
qlearning.load(join(_CURRENT_DIR, '..', 'artifacts', 'qlearning'))

rewards = []
for i in range(EPISODES):
    state = env.reset()
    _reward = 0
    n = 0
    done = False
    with tqdm(desc=f'Episode {i + 1}') as pbar:
        while not done:
            n += 1

            action = qlearning.take_action(state)
            action = actions_map[action]

            state, reward, done, info = env.step(action)
            qlearning.update(state, reward)

            _reward += reward

            pbar.update(1)
            pbar.set_postfix({'Cumulative reward': _reward})

    rewards.append(_reward)
    qlearning.save(join(_CURRENT_DIR, '..', 'artifacts', 'qlearning'))

env.close()

plt.plot(rewards)
plt.xlabel('Episodes')
plt.ylabel('Cumulative Reward')
plt.title('Evolution of Cumulative Reward')
plt.savefig('Q_learning_agent')
plt.show()
