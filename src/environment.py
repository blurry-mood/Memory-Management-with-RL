from copy import deepcopy
from typing import List, Union, Any

import gym
import numpy as np
from gym import Env
from numpy import ndarray


class MemoryEnvironment(Env):

    def __init__(self, memory: List[int] = None, arrays: List[int] = None, memory_size: int = -1, n_arrays: int = -1,
                 max_size: int = 10 ** 3):
        super(MemoryEnvironment, self).__init__()

        assert (memory is not None and arrays is not None) or (
                memory_size > 0 and n_arrays > 0 and max_size > 0), "Memory and Arrays must be provided, or memory_size, n_arrays, and max_size must be provided"

        self.i = 0
        self.memory_size = memory_size
        self.memory = memory
        self.arrays = arrays
        self.n_arrays = n_arrays
        self.max_size = max_size

        if self.memory_size <= 0:
            self.action_space = gym.spaces.Discrete(len(memory))
            self.observation_space = gym.spaces.Box(low=0, high=len(arrays), shape=(len(memory),))
        else:
            self.action_space = gym.spaces.Discrete(memory_size)
            self.observation_space = gym.spaces.Box(low=0, high=n_arrays, shape=(memory_size,))

    def reset(self):
        self.i = 0
        if self.memory_size > 0:
            self.memory = np.random.randint(low=1, high=self.max_size, size=(self.memory_size,))
            self.arrays = np.random.randint(low=1, high=self.max_size, size=(self.n_arrays,))
        else:
            self.memory = np.array(self.memory)
            self.arrays = np.array(self.arrays)
        return self._state()

    def _state(self):
        if self.i == len(self.arrays):
            return None
        return deepcopy(np.concatenate((self.memory, self.arrays), axis=0))

    def step(self, action: int) -> Union[
        tuple[None, float, bool, dict[Any, Any]], tuple[ndarray, int, bool, dict[Any, Any]]]:
        if self.arrays[self.i] > self.memory[action]:
            reward = -1  # meaning that this designated allocation will overwrite its neighboring cells
        else:
            reward = float(np.mean(self.memory == 0))

        self.memory[action] = self.memory[action] - self.arrays[self.i]
        self.arrays[self.i] = 0

        self.i += 1
        return self._state(), self.i / len(self.arrays) * reward + 0.9*float(np.mean(self.memory>=0)) , self.i == len(self.arrays), {}

    def render(self, mode='human'):
        print(f"\n\n** Step: {self.i + 1}")
        print("Current Memory Grid:", self.memory)
        print("Current Arrays Status:", self.arrays)

    def close(self):
        super(MemoryEnvironment, self).close()
