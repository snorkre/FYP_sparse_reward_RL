from collections import deque
import random
import numpy as np


class ReplayBuffer:
    """ 
    Experience replay buffer for storing and sampling past transitions.
    
    Enables off-policy learning by breaking temporal correlations between
    consecutive experiences and allows for efficient reuse of past data.
    """
    def __init__(self, capacity: int, seed: int = 0):
        self.buffer = deque(maxlen=capacity)
        self.rng = random.Random(seed)

    def __len__(self):
        """
        Returns the current number of transitions.
        Used to check if enough data is available before training starts.
        """
        return len(self.buffer)

    def push(self, s, a, r, s2, done):
        """
        Stores a single transition tuple (s, a, r, s', done).
        These transitions are later sampled in batches to agent.
        """
        self.buffer.append((s, a, r, s2, done))

    def sample(self, batch_size: int):
        """
        Samples a random mini-batch of transitions.
        
        Random sampling:
        - Reduces correlation between sequential experiences
        - Stabilises training compared to online updated
        """
        batch = self.rng.sample(self.buffer, batch_size)

        # Unzip batch into separate components
        s, a, r, s2, done = zip(*batch)

        # Convert lists to numpy arrays for efficient batch processing in PyTorch
        return (
            np.array(s, dtype=np.float32),
            np.array(a, dtype=np.int64),
            np.array(r, dtype=np.float32),
            np.array(s2, dtype=np.float32),
            np.array(done, dtype=np.float32),
        )
