from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class QNetwork(nn.Module):
    """
    feedforward neural network used to approximate the Q-function
    Maps a state (observation) to Q-values for all possible actions
    """

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128) -> None: 
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(), 
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear (hidden, n_actions),
        )
            
               

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ 
        forward pass: returns Q-values for all actions given input state(s)
        """
        return self.net(x)
    

@dataclass
class DQNConfig:
    """
    Deep Q-network agent with Double DQN enhancement.
    uses:
    -online network (q) for action selection and learning
    -target network (q_tgt) for stability."""
    gamma: float = 0.99
    lr: float = 1e-3
    batch_size: int = 64
    buffer_size: int = 50_000
    min_buffer: int = 2_000
    target_update_every: int = 1_000
    grad_clip_norm: float = 10.0

class DQNAgent:
    def __init__(self, obs_dim: int, n_actions:int, device: torch.device, cfg: DQNConfig, seed: int = 0) -> None:

        # set random seeds for reproducibility across runs 
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        self.device = device
        self.cfg = cfg
        self.n_action = n_actions
        self.step_count = 0

        # Main Q-network (updated every step)
        self.q = QNetwork (obs_dim, n_actions).to(device)

        # Target network (updated frequently for stability)
        self.q_tgt = QNetwork(obs_dim, n_actions).to(device)
        self.q_tgt.load_state_dict(self.q.state_dict())
        self.q_tgt.eval() # target network is not trained directly

        self.optim = optim.Adam(self.q.parameters(), lr=cfg.lr)

        # Huber loss is more robust to outliers than MSE, improving stability
        self.loss_fn = nn.SmoothL1Loss() #huber


    @torch.no_grad()
    def act(self, obs: np.ndarray, eps: float) -> int:
        """
        Selected an action using epsilon-greedy policy.
        With probability epsilon: explore (random action)
        Otherwise: exploit (action with highest Q-value from online network)"""
        if np.random.rand() < eps:
            return np.random.randint(self.n_action)
        x = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        q = self.q(x)
        return int(torch.argmax(q, dim=1).item())
    
    def update_target_if_needed(self) -> None:
        """
        Periodically update the target network to match the online network for stability.
        This stabilizes learning by preventing rapid shifting targets
        """
        if self.step_count % self.cfg.target_update_every == 0:
            self.q_tgt.load_state_dict(self.q.state_dict())

    def train_step(self, batch) -> float: 
        """
        Performs a single training step on a batch of transitions sampled from the replay buffer.
        Implements the Double DQN:
        - Online network selects the best action
        - Target network evaluates that action"""
        s, a, r, s2, done = batch

        # Convert batch data to tensors
        s = torch.tensor(s, dtype=torch.float32, device=self.device)
        a = torch.tensor(a, dtype=torch.int64, device=self.device). unsqueeze(1)
        r = torch.tensor(r, dtype=torch.float32, device=self.device).unsqueeze(1)
        s2 = torch.tensor(s2, dtype=torch.float32, device=self.device)
        done = torch.tensor(done, dtype=torch.float32, device=self.device).unsqueeze(1)

        # Q(s, a) predicted Q value for the action taken in the batch
        q_sa = self.q(s).gather(1, a) 

        #Double DQN target
        with torch.no_grad():
            # Action selection from online network
            a2 = torch.argmax(self.q(s2), dim=1, keepdim=True)

            # Action evaluation from target network
            q_s2a2 = self.q_tgt(s2).gather(1, a2)
            
            # Bellman target
            y = r + (1.0 - done) * self.cfg.gamma * q_s2a2

        # Compute loss between predicted Q-values and target Q-values
        loss = self.loss_fn(q_sa, y)

        #Backpropagation
        self.optim.zero_grad(set_to_none=True)
        loss.backward()

        # Gradient clipping prevents exploding gradients
        nn.utils.clip_grad_norm_(self.q.parameters(), self.cfg.grad_clip_norm)
        self.optim.step()

        return float(loss.item())
    

        
        

