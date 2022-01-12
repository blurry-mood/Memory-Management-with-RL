from environment import MemoryEnvironment
from os.path import split, join
from rl_algorithms import QLearning

_CURRENT_DIR = split(__file__)[0]

ALPHA = 2e-1
GAMMA = 0.99
EPS = 1e-1
ITERS = 2


class MemoryAgent(QLearning):
    def decode_state(self, state):
        if state is None:
            return None
        return tuple(state)


env = MemoryEnvironment(memory_size=4, n_arrays=10)

qlearning = MemoryAgent(actions=list(range(env.action_space.n)), alpha=ALPHA, gamma=GAMMA, eps=EPS)
qlearning.load(join(_CURRENT_DIR, '..', 'artifacts', 'qlearning'))

for i in range(ITERS):
    state = env.reset()
    env.render()
    n = 0
    done = False
    while not done:
        n += 1
        action = qlearning.take_action(state)
        state, reward, done, info = env.step(action)
        qlearning.update(state, reward)
        env.render()

    print(f'*********** Episode {i} finished after {n} steps, with a reward equal to {reward}.\n\n')
    qlearning.save(join(_CURRENT_DIR, '..', 'artifacts', 'qlearning'))

env.close()
