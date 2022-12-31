from collections import deque, namedtuple
import random

# For comodity I'll use this named tuple to store the agent's memory
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        """Sample a random batch size from the memory"""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        """Returns the size of the memory"""
        return len(self.memory)