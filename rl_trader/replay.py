"""
Experience Replay Buffers
=========================
Uniform sampling replay buffer and Prioritized Experience Replay (PER)
with SumTree data structure for O(log N) sampling and updates.
"""

import numpy as np
from typing import Tuple, List, Optional


class SumTree:
    """Binary sum-tree for efficient prioritized sampling.

    Leaf nodes store priorities; internal nodes store sums.
    Supports O(log N) sampling proportional to priority and O(log N) updates.
    """
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.n_entries = 0

    def _propagate(self, idx: int, change: float):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, s: float) -> int:
        left = 2 * idx + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        return self._retrieve(left + 1, s - self.tree[left])

    def total(self) -> float:
        return self.tree[0]

    def add(self, priority: float, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, priority)
        self.write = (self.write + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, idx: int, priority: float):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, s: float) -> Tuple[int, float, object]:
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]


class ReplayBuffer:
    """Uniform random experience replay buffer."""

    def __init__(self, capacity: int = 100000):
        self.capacity = capacity
        self.buffer: List[Tuple] = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for i in indices:
            s, a, r, ns, d = self.buffer[i]
            states.append(s)
            actions.append(a)
            rewards.append(r)
            next_states.append(ns)
            dones.append(d)
        weights = np.ones(batch_size, dtype=np.float32)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
            weights,
            indices,
        )

    def update_priorities(self, indices, td_errors):
        """No-op for uniform replay."""
        pass

    def __len__(self) -> int:
        return len(self.buffer)


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay with proportional prioritization.

    Uses SumTree for O(log N) sampling and updates.
    Priority p_i = (|td_error| + epsilon) ** alpha
    Weight w_i = (N * P(i)) ** (-beta), normalized by max weight.
    """
    def __init__(self, capacity: int = 100000, alpha: float = 0.6,
                 beta: float = 0.4, beta_increment: float = 0.001,
                 epsilon: float = 1e-6):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        self.max_priority = 1.0

    def push(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        self.tree.add(self.max_priority ** self.alpha, experience)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        batch_size = min(batch_size, self.tree.n_entries)
        indices = np.zeros(batch_size, dtype=np.int32)
        priorities = np.zeros(batch_size, dtype=np.float32)
        states, actions, rewards, next_states, dones = [], [], [], [], []

        segment = self.tree.total() / batch_size
        self.beta = min(1.0, self.beta + self.beta_increment)

        for i in range(batch_size):
            s_val = np.random.uniform(segment * i, segment * (i + 1))
            idx, p, data = self.tree.get(s_val)
            indices[i] = idx
            priorities[i] = p
            s, a, r, ns, d = data
            states.append(s)
            actions.append(a)
            rewards.append(r)
            next_states.append(ns)
            dones.append(d)

        # Importance sampling weights
        prob = priorities / self.tree.total()
        weights = (self.tree.n_entries * prob) ** (-self.beta)
        weights /= weights.max()

        return (
            np.array(states, dtype=np.float32),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
            weights.astype(np.float32),
            indices,
        )

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(float(td_error)) + self.epsilon) ** self.alpha
            self.tree.update(int(idx), priority)
            self.max_priority = max(self.max_priority, priority)

    def __len__(self) -> int:
        return self.tree.n_entries
