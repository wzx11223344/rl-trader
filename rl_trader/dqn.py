"""
DQN Agent with Double DQN, Dueling Architecture, and Prioritized Experience Replay
==================================================================================
Pure NumPy implementation. No PyTorch/TensorFlow. No stable-baselines/gym.
"""

import numpy as np
import os
import pickle
from typing import Tuple, Dict, List, Optional
from .networks import Dense
from .replay import ReplayBuffer, PrioritizedReplayBuffer


class DQNNetwork:
    """Q-Network built from scratch NumPy layers.

    Supports standard DQN and Dueling architecture (value + advantage streams).
    """
    def __init__(self, state_dim: int, action_dim: int,
                 hidden_layers: List[int] = None,
                 dueling: bool = False, lr: float = 0.001):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.dueling = dueling
        if hidden_layers is None:
            hidden_layers = [128, 128]

        self.layers = []
        prev_dim = state_dim
        for i, units in enumerate(hidden_layers):
            self.layers.append(Dense(units, activation="relu",
                                     input_dim=prev_dim, name=f"dense_{i}"))
            prev_dim = units

        if dueling:
            # Value stream
            self.value_layer = Dense(hidden_layers[-1] // 2, activation="relu",
                                     input_dim=prev_dim, name="value_hidden")
            self.value_out = Dense(1, activation="linear",
                                   input_dim=hidden_layers[-1] // 2, name="value_out")
            # Advantage stream
            self.adv_layer = Dense(hidden_layers[-1] // 2, activation="relu",
                                   input_dim=prev_dim, name="adv_hidden")
            self.adv_out = Dense(action_dim, activation="linear",
                                 input_dim=hidden_layers[-1] // 2, name="adv_out")
        else:
            self.output_layer = Dense(action_dim, activation="linear",
                                      input_dim=prev_dim, name="output")

        self.lr = lr

        # Pre-build with a dummy forward pass so weights are initialized
        x = np.random.randn(1, state_dim).astype(np.float32)
        self.forward(x)

    def forward(self, x: np.ndarray) -> np.ndarray:
        h = x
        for layer in self.layers:
            h = layer.forward(h)
        if self.dueling:
            v = self.value_layer.forward(h)
            v = self.value_out.forward(v)  # (B, 1)
            a = self.adv_layer.forward(h)
            a = self.adv_out.forward(a)    # (B, A)
            return v + a - np.mean(a, axis=-1, keepdims=True)
        return self.output_layer.forward(h)

    def backward(self, dout: np.ndarray):
        if self.dueling:
            # Q(s,a) = V(s) + A(s,a) - mean(A(s))
            # dL/dV = sum(dL/dQ) over actions -> (B, 1)
            # dL/dA_j = dL/dQ_j - mean(dL/dQ) -> (B, A)
            dv_grad = np.sum(dout, axis=-1, keepdims=True)  # (B, 1)
            da_grad = dout - np.mean(dout, axis=-1, keepdims=True)  # (B, A)
            da_grad = self.adv_out.backward(da_grad)
            da_grad = self.adv_layer.backward(da_grad)
            dv_grad = self.value_out.backward(dv_grad)
            dv_grad = self.value_layer.backward(dv_grad)
            d_hidden = da_grad + dv_grad
        else:
            d_hidden = self.output_layer.backward(dout)

        for layer in reversed(self.layers):
            d_hidden = layer.backward(d_hidden)

    def get_params(self) -> Dict:
        params = {"state_dim": self.state_dim, "action_dim": self.action_dim,
                  "dueling": self.dueling}
        for i, layer in enumerate(self.layers):
            params[f"layer_{i}_W"] = layer.W.copy()
            params[f"layer_{i}_b"] = layer.b.copy()
        if self.dueling:
            for name in ["value_layer", "value_out", "adv_layer", "adv_out"]:
                layer = getattr(self, name)
                params[f"{name}_W"] = layer.W.copy()
                params[f"{name}_b"] = layer.b.copy()
        else:
            params["output_W"] = self.output_layer.W.copy()
            params["output_b"] = self.output_layer.b.copy()
        return params

    def set_params(self, params: Dict):
        for i, layer in enumerate(self.layers):
            layer.W = params[f"layer_{i}_W"].copy()
            layer.b = params[f"layer_{i}_b"].copy()
        if self.dueling:
            for name in ["value_layer", "value_out", "adv_layer", "adv_out"]:
                layer = getattr(self, name)
                layer.W = params[f"{name}_W"].copy()
                layer.b = params[f"{name}_b"].copy()
        else:
            self.output_layer.W = params["output_W"].copy()
            self.output_layer.b = params["output_b"].copy()


class DQNAgent:
    """DQN Agent with Double DQN, Dueling Architecture, and PER.

    Args:
        state_dim: Dimension of state space.
        action_dim: Number of discrete actions.
        hidden_layers: List of hidden layer sizes.
        lr: Learning rate.
        gamma: Discount factor.
        epsilon_start/end/decay: Epsilon-greedy exploration parameters.
        tau: Soft update rate for target network.
        buffer_capacity: Replay buffer size.
        batch_size: Training batch size.
        use_PER: Whether to use Prioritized Experience Replay.
        use_double: Whether to use Double DQN.
        use_dueling: Whether to use Dueling architecture.
        n_step: N-step returns (default 3).
    """
    def __init__(self, state_dim: int, action_dim: int,
                 hidden_layers: List[int] = None,
                 lr: float = 0.001, gamma: float = 0.99,
                 epsilon_start: float = 1.0, epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.995, tau: float = 0.005,
                 buffer_capacity: int = 100000, batch_size: int = 64,
                 use_PER: bool = True, use_double: bool = True,
                 use_dueling: bool = True, n_step: int = 3):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.tau = tau
        self.batch_size = batch_size
        self.use_double = use_double
        self.n_step = n_step

        # Online and target networks
        self.q_network = DQNNetwork(state_dim, action_dim, hidden_layers,
                                    dueling=use_dueling, lr=lr)
        self.target_network = DQNNetwork(state_dim, action_dim, hidden_layers,
                                         dueling=use_dueling, lr=lr)
        self._hard_update()

        # Replay buffer
        if use_PER:
            self.memory = PrioritizedReplayBuffer(capacity=buffer_capacity,
                                                   alpha=0.6, beta=0.4)
        else:
            self.memory = ReplayBuffer(capacity=buffer_capacity)

        # N-step buffer
        self.n_step_buffer: List[Tuple] = []

        self.train_step = 0

    def _hard_update(self):
        """Copy online network weights to target network."""
        self.target_network.set_params(self.q_network.get_params())

    def _soft_update(self):
        """Soft update: target = tau * online + (1 - tau) * target"""
        q_params = self.q_network.get_params()
        t_params = self.target_network.get_params()
        for key in q_params:
            t_params[key] = self.tau * q_params[key] + (1 - self.tau) * t_params[key]
        self.target_network.set_params(t_params)

    def act(self, state: np.ndarray, training: bool = True) -> int:
        """Select action using epsilon-greedy policy."""
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)

        state_batch = state.reshape(1, -1)
        q_values = self.q_network.forward(state_batch)
        return int(np.argmax(q_values[0]))

    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """Return Q-values for all actions."""
        return self.q_network.forward(state.reshape(1, -1))[0]

    def push(self, state, action, reward, next_state, done):
        """Store transition with n-step return handling."""
        self.n_step_buffer.append((state, action, reward, next_state, done))

        if len(self.n_step_buffer) < self.n_step:
            return

        # Compute n-step return
        n_reward = 0.0
        for i in range(self.n_step):
            n_reward += (self.gamma ** i) * self.n_step_buffer[i][2]

        s = self.n_step_buffer[0][0]
        a = self.n_step_buffer[0][1]
        ns = self.n_step_buffer[-1][3]
        d = self.n_step_buffer[-1][4]

        # Add bootstrap value if n_step_buffer[-1] is not terminal
        if not d:
            ns_batch = ns.reshape(1, -1)
            n_reward += (self.gamma ** self.n_step) * np.max(
                self.target_network.forward(ns_batch)[0])

        self.memory.push(s, a, n_reward, ns, d)
        self.n_step_buffer.pop(0)

    def flush_n_step(self):
        """Flush remaining n-step buffer at episode end."""
        while len(self.n_step_buffer) > 0:
            s = self.n_step_buffer[0][0]
            a = self.n_step_buffer[0][1]
            ns = self.n_step_buffer[-1][3]
            d = True  # Force terminal
            r = sum((self.gamma ** i) * self.n_step_buffer[i][2]
                    for i in range(len(self.n_step_buffer)))
            self.memory.push(s, a, r, ns, d)
            self.n_step_buffer.pop(0)

    def update(self) -> Optional[Dict]:
        """Perform one training step. Returns loss info dict."""
        if len(self.memory) < self.batch_size:
            return None

        self.train_step += 1
        states, actions, rewards, next_states, dones, weights, indices = \
            self.memory.sample(self.batch_size)
        actions = np.array(actions, dtype=np.int32)

        # Current Q values
        q_values = self.q_network.forward(states)
        current_q = q_values[np.arange(self.batch_size), actions]

        # Target Q values
        if self.use_double:
            # Double DQN: online selects action, target evaluates
            next_q_online = self.q_network.forward(next_states)
            best_actions = np.argmax(next_q_online, axis=1)
            next_q_target = self.target_network.forward(next_states)
            next_q = next_q_target[np.arange(self.batch_size), best_actions]
        else:
            next_q = np.max(self.target_network.forward(next_states), axis=1)

        target_q = rewards + self.gamma * next_q * (1 - dones)

        # TD errors for PER update
        td_errors = target_q - current_q

        # Gradient: dL/dq = 2 * (q - target) * weights^2 / batch_size
        grad = 2.0 * (current_q - target_q) * (weights ** 2) / self.batch_size
        grad_full = np.zeros_like(q_values)
        grad_full[np.arange(self.batch_size), actions] = grad
        self.q_network.backward(grad_full)

        # Update priorities
        self.memory.update_priorities(indices, td_errors)

        # Soft update target network
        self._soft_update()

        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        return {"loss": float(np.mean((td_errors * weights) ** 2)),
                "epsilon": self.epsilon, "mean_q": float(np.mean(current_q))}

    def save(self, path: str):
        """Save agent weights."""
        data = {
            "q_network": self.q_network.get_params(),
            "target_network": self.target_network.get_params(),
            "epsilon": self.epsilon,
            "train_step": self.train_step,
        }
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def load(self, path: str):
        """Load agent weights."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.q_network.set_params(data["q_network"])
        self.target_network.set_params(data["target_network"])
        self.epsilon = data.get("epsilon", self.epsilon_end)
        self.train_step = data.get("train_step", 0)
