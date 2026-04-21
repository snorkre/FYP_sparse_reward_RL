from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class QNetwork(nn.Module):
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
        return self.net(x)
    

@dataclass
class DQNConfig:
    gamma: float = 0.99
    lr: float = 1e-3
    batch_size: int = 64
    buffer_size: int = 50_000
    min_buffer: int = 2_000
    target_update_every: int = 1_000
    grad_clip_norm: float = 10.0

class DQNAgent:
    def __init__(self, obs_dim: int, n_actions:int, device: torch.device, cfg: DQNConfig, seed: int = 0) -> None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        self.device = device
        self.cfg = cfg
        self.n_action = n_actions
        self.step_count = 0

        self.q = QNetwork (obs_dim, n_actions).to(device)
        self.q_tgt = QNetwork(obs_dim, n_actions).to(device)
        self.q_tgt.load_state_dict(self.q.state_dict())
        self.q_tgt.eval()

        self.optim = optim.Adam(self.q.parameters(), lr=cfg.lr)
        self.loss_fn = nn.SmoothL1Loss() #huber


    @torch.no_grad()
    def act(self, obs: np.ndarray, eps: float) -> int:
        if np.random.rand() < eps:
            return np.random.randint(self.n_action)
        x = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        q = self.q(x)
        return int(torch.argmax(q, dim=1).item())
    
    def update_target_if_needed(self) -> None:
        if self.step_count % self.cfg.target_update_every == 0:
            self.q_tgt.load_state_dict(self.q.state_dict())

    def train_step(self, batch) -> float: 
        s, a, r, s2, done = batch
        s = torch.tensor(s, dtype=torch.float32, device=self.device)
        a = torch.tensor(a, dtype=torch.int64, device=self.device). unsqueeze(1)
        r = torch.tensor(r, dtype=torch.float32, device=self.device).unsqueeze(1)
        s2 = torch.tensor(s2, dtype=torch.float32, device=self.device)
        done = torch.tensor(done, dtype=torch.float32, device=self.device).unsqueeze(1)

        q_sa = self.q(s).gather(1, a) # Q value for action taken in batch

        #Double DQN target
        with torch.no_grad():
            a2 = torch.argmax(self.q(s2), dim=1, keepdim=True)
            q_s2a2 = self.q_tgt(s2).gather(1, a2)
            y = r + (1.0 - done) * self.cfg.gamma * q_s2a2

        loss = self.loss_fn(q_sa, y)

        self.optim.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), self.cfg.grad_clip_norm)
        self.optim.step()

        return float(loss.item())
    

        
        

