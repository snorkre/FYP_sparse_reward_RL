from collections import deque
import random
import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int = 0):
        self.buffer = deque(maxlen=capacity)
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.buffer)

    def push(self, s, a, r, s2, done):
        self.buffer.append((s, a, r, s2, done))

    def sample(self, batch_size: int):
        batch = self.rng.sample(self.buffer, batch_size)
        s, a, r, s2, done = zip(*batch)

        return (
            np.array(s, dtype=np.float32),
            np.array(a, dtype=np.int64),
            np.array(r, dtype=np.float32),
            np.array(s2, dtype=np.float32),
            np.array(done, dtype=np.float32),
        )
