from typing import List, Tuple

import numpy as np
from gym import Env

from config import *


class MemoryEnvironment(Env):

    def __init__(self, n_slots: int = -1, n_arrays: int = -1, max_size: int = 10 ** 3):
        super(MemoryEnvironment, self).__init__()

        assert n_slots > 0 and n_arrays > 0 and max_size > 0, "n_slots, n_arrays, and max_size must be positive integers"

        self.arrays = None
        self.memory = None

        self.n_slots = n_slots
        self.n_arrays = n_arrays
        self.max_size = max_size

        self.action_space = None
        self.observation_space = None

    def reset(self):
        self.memory: List = np.random.randint(low=-self.max_size, high=self.max_size, size=(self.n_slots,)).tolist()
        self.arrays: List = np.random.randint(low=0, high=self.max_size, size=(self.n_arrays,)).tolist()
        return self._state_done()[0]

    def _compress_state(self):
        # remove empty arrays, i.e. moved to a memory slot
        for i in range(len(self.arrays)):
            try:
                self.arrays.remove(0)
            except:
                pass
        # assemble memory blocks that are free
        for i in range(len(self.memory)):
            j = i
            summ = 0
            while j < len(self.memory) and self.memory[j] > 0:
                summ += self.memory[j]
                self.memory[j] = 0
                j += 1

            if i < j:
                self.memory[i] = summ

        # remove empty memory slots
        for i in range(len(self.memory)):
            try:
                self.memory.remove(0)
            except:
                pass

    def _state_done(self):
        if self.arrays == []:
            return None, True
        self._compress_state()
        return (tuple(self.memory), tuple(self.arrays)), False

    def step(self, action: Tuple[int, int, int]):
        """

        Args:
            action = (array_index, slot_index, direction)
            If direction is 0, then the array is moved to the slot.
            If direction is 1, then the slot is moved to the array.

        """
        array, slot, direction = action
        assert direction in [0, 1], f"direction must be 0 or 1"

        if direction == 0:
            if array >= len(self.arrays):  # if the chosen index doesn't map to an existing array in arrays list
                reward = INVALID_ACTION_REWARD
            else:
                if slot >= len(self.memory):  # if the agent chooses to allocate memory after the last block in memory
                    self.memory.append(-self.arrays[array])
                    self.arrays[array] = 0
                    reward = STEP_REWARD
                elif self.arrays[array] > self.memory[
                    slot]:  # if the array size is larger than the free memory, append it at the end
                    self.memory.append(-self.arrays[array])
                    reward = INVALID_ACTION_REWARD
                    self.arrays[array] = 0
                else:  # otherwise perform the action
                    self.memory[slot] -= self.arrays[array]
                    self.memory.insert(slot, -self.arrays[array])
                    self.arrays[array] = 0
                    reward = STEP_REWARD
        else:
            if slot >= len(self.memory):
                reward = INVALID_ACTION_REWARD
            elif self.memory[slot] >= 0:  # make sure there's an allocated array in that location
                reward = INVALID_ACTION_REWARD
            else:
                self.arrays.append(-self.memory[slot])
                self.memory[slot] = -self.memory[slot]
                reward = -MEMORY_TO_ARRAY_REWARD * self.memory[slot]
        state, done = self._state_done()
        if done:
            reward += END_EPISODE_REWARD / np.sum(np.array(self.memory) > 0)

        return state, reward, done, {}

    def render(self, mode='human'):
        print(f"\n** Step")
        print("Current Memory Grid:", self.memory)
        print("Current Arrays Status:", self.arrays)


def close(self):
    super(MemoryEnvironment, self).close()
