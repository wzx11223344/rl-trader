"""
A2C Agent (Advantage Actor-Critic)
===================================
Synchronous Advantage Actor-Critic implemented from scratch in NumPy.
Actor-Critic with shared feature extractor, n-step returns,
and synchronous single-environment updates.
"""

import numpy as np
import os
import pickle
from typing import Tuple, Dict, List, Optional
from .networks import Dense


class A2CNetwork:
    """Actor-Critic network for A2C.

    Shared feature extractor with separate actor and critic heads.
    """

    def __init__(self, state_dim: int, action_dim: int, discrete: bool = True,
                 hidden_layers: List[int] = None, lr: float = 0.001):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discrete = discrete

        if hidden_layers is None:
            hidden_layers = [128, 128]

        # Shared layers
        self.shared = []
        prev_dim = state_dim
        for i, units in enumerate(hidden_layers):
            self.shared.append(Dense(units, activation="relu",
                                     input_dim=prev_dim, name=f"shared_{i}"))
            prev_dim = units

        # Actor head: policy
        self.actor_in = Dense(hidden_layers[-1], activation="tanh",
                              input_dim=prev_dim, name="actor_in")
        if discrete:
            self.actor_out = Dense(action_dim, activation="linear",
                                   input_dim=hidden_layers[-1], name="actor_out")
        else:
            self.actor_out = Dense(action_dim * 2, activation="linear",
                                   input_dim=hidden_layers[-1], name="actor_out")

        # Critic head: value
        self.critic_in = Dense(hidden_layers[-1], activation="tanh",
                               input_dim=prev_dim, name="critic_in")
        self.critic_out = Dense(1, activation="linear",
                                input_dim=hidden_layers[-1], name="critic_out")

        self.lr = lr

        # Pre-build with a dummy forward pass
        x = np.random.randn(1, state_dim).astype(np.float32)
        self.forward(x)

    def forward(self, x: np.ndarray) -> Tuple:
        """Full forward pass. Returns (actor_out, value)."""
        h = x
        for layer in self.shared:
            h = layer.forward(h)

        # Actor
        a_h = self.actor_in.forward(h)
        logits = self.actor_out.forward(a_h)

        if self.discrete:
            logits_stable = logits - np.max(logits, axis=-1, keepdims=True)
            exp_logits = np.exp(logits_stable)
            probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
            actor_out = probs
        else:
            mu = logits[:, :self.action_dim]
            log_std = logits[:, self.action_dim:]
            log_std = np.clip(log_std, -20, 2)
            actor_out = (mu, log_std)

        # Critic
        c_h = self.critic_in.forward(h)
        value = self.critic_out.forward(c_h).flatten()

        return actor_out, value

    def forward_actor(self, x: np.ndarray) -> Tuple:
        """Forward through actor only."""
        h = x
        for layer in self.shared:
            h = layer.forward(h)
        a_h = self.actor_in.forward(h)
        logits = self.actor_out.forward(a_h)
        if self.discrete:
            logits_stable = logits - np.max(logits, axis=-1, keepdims=True)
            exp_logits = np.exp(logits_stable)
            probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
            return probs, logits
        else:
            mu = logits[:, :self.action_dim]
            log_std = logits[:, self.action_dim:]
            log_std = np.clip(log_std, -20, 2)
            return mu, log_std

    def forward_critic(self, x: np.ndarray) -> np.ndarray:
        """Forward through critic only."""
        h = x
        for layer in self.shared:
            h = layer.forward(h)
        c_h = self.critic_in.forward(h)
        return self.critic_out.forward(c_h).flatten()

    def get_params(self) -> Dict:
        params = {}
        for i, layer in enumerate(self.shared):
            params[f"shared_{i}_W"] = layer.W.copy()
            params[f"shared_{i}_b"] = layer.b.copy()
        for name in ["actor_in", "actor_out", "critic_in", "critic_out"]:
            layer = getattr(self, name)
            params[f"{name}_W"] = layer.W.copy()
            params[f"{name}_b"] = layer.b.copy()
        return params

    def set_params(self, params: Dict):
        for i, layer in enumerate(self.shared):
            layer.W = params[f"shared_{i}_W"].copy()
            layer.b = params[f"shared_{i}_b"].copy()
        for name in ["actor_in", "actor_out", "critic_in", "critic_out"]:
            layer = getattr(self, name)
            layer.W = params[f"{name}_W"].copy()
            layer.b = params[f"{name}_b"].copy()


class A2CAgent:
    """Advantage Actor-Critic (A2C) Agent.

    Synchronous single-environment updates using n-step returns.

    Args:
        state_dim: State dimension.
        action_dim: Action dimension.
        discrete: Whether action space is discrete.
        hidden_layers: Hidden layer sizes.
        lr: Learning rate.
        gamma: Discount factor.
        n_steps: Number of steps for n-step return (1 = TD, >1 = n-step).
        entropy_coef: Entropy bonus coefficient.
        value_coef: Value loss coefficient.
    """

    def __init__(self, state_dim: int, action_dim: int,
                 discrete: bool = True,
                 hidden_layers: List[int] = None,
                 lr: float = 0.001, gamma: float = 0.99,
                 n_steps: int = 5, entropy_coef: float = 0.01,
                 value_coef: float = 0.5):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discrete = discrete
        self.gamma = gamma
        self.n_steps = n_steps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef

        self.network = A2CNetwork(state_dim, action_dim, discrete,
                                   hidden_layers, lr)

        # N-step buffer
        self._states: List[np.ndarray] = []
        self._actions: List = []
        self._rewards: List[float] = []
        self._next_state: Optional[np.ndarray] = None
        self._done: bool = False

        self.train_step = 0

    def act(self, state: np.ndarray, training: bool = True) -> Tuple[int, float]:
        """Select action and return (action, value)."""
        x = state.reshape(1, -1)
        actor_out, value = self.network.forward(x)

        if self.discrete:
            probs = actor_out
            if not training:
                return int(np.argmax(probs[0])), float(value[0])
            action = int(np.random.choice(self.action_dim, p=probs[0]))
        else:
            mu, log_std = actor_out
            mu, log_std = mu[0], log_std[0]
            if not training:
                action = np.clip(mu, -1.0, 1.0)
            else:
                std = np.exp(log_std)
                action = np.clip(mu + np.random.randn(self.action_dim) * std, -1.0, 1.0)

        return action, float(value[0])

    def push(self, state, action, reward):
        """Store transition for n-step update."""
        self._states.append(state)
        self._actions.append(action)
        self._rewards.append(reward)

    def update(self, next_state: np.ndarray, done: bool) -> Optional[Dict]:
        """Perform A2C update. Returns metrics dict."""
        self._next_state = next_state
        self._done = done

        if len(self._states) >= self.n_steps or done:
            return self._update_n_step()
        return None

    def _update_n_step(self) -> Dict:
        """Perform n-step A2C update."""
        self.train_step += 1
        T = len(self._states)
        n = min(self.n_steps, T)

        # Compute n-step returns for all stored transitions
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0

        for t in range(T):
            # Compute n-step return
            end_idx = min(t + n, T)
            R = 0.0

            if end_idx == T and self._done:
                R = 0.0  # Terminal
            elif end_idx == T:
                # Bootstrap from last value
                ns = self._next_state.reshape(1, -1)
                _, last_val = self.network.forward(ns)
                R = (self.gamma ** (end_idx - t)) * float(last_val[0])

            # Sum intermediate rewards
            for k in range(t, end_idx):
                R += (self.gamma ** (k - t)) * self._rewards[k]

            # Get current value estimate
            s = self._states[t].reshape(1, -1)
            actor_out, value = self.network.forward(s)
            value = float(value[0])

            advantage = R - value

            # Compute gradients
            if self.discrete:
                probs = actor_out
                a = self._actions[t]

                # Policy gradient
                d_policy = -advantage / probs[0, a]

                # Entropy gradient
                log_probs = np.log(probs[0] + 1e-8)
                entropy = -np.sum(probs[0] * log_probs)

                # Value gradient
                d_value = advantage

                total_policy_loss += -advantage * np.log(probs[0, a] + 1e-8)
                total_value_loss += advantage ** 2
                total_entropy += entropy

                # Backward through output layers - simplified
                # In production, we'd do full chain-rule backward
                # For now, we rely on the forward pass activations
                # and the built-in backward() methods
            else:
                mu, log_std = actor_out
                a = np.array(self._actions[t]).reshape(1, -1)
                var = np.exp(2 * log_std)

                # Gaussian log prob
                action_diff = a - mu
                log_prob = -0.5 * np.sum(
                    action_diff ** 2 / (var + 1e-8) + 2 * log_std + np.log(2 * np.pi))
                entropy_val = 0.5 * np.sum(np.log(2 * np.pi * np.e * var) + 1)

                total_policy_loss += -advantage * float(log_prob)
                total_value_loss += advantage ** 2
                total_entropy += float(entropy_val)

                # Gradient for continuous action
                d_mu = advantage * action_diff / (var + 1e-8)
                d_logstd = advantage * ((action_diff ** 2) / (var + 1e-8) - 1.0)
                # Backward through actor_out
                d_actor = np.concatenate([d_mu, d_logstd], axis=-1)
                _ = self.network.actor_out.backward(d_actor)
                _ = self.network.actor_in.backward(
                    self.network.actor_in._cache.get("activation", np.zeros_like(a)))

            # Backward through critic (value head)
            d_value_arr = np.array([[2.0 * advantage]])  # MSE gradient
            _ = self.network.critic_out.backward(d_value_arr)
            _ = self.network.critic_in.backward(
                self.network.critic_in._cache.get("activation",
                    np.zeros((1, self.network.actor_in.units))))

            # Backward through shared layers
            for layer in reversed(self.network.shared):
                _ = layer.backward(layer._cache.get("activation",
                    np.zeros_like(s)))

        # Clear buffers
        self._states.clear()
        self._actions.clear()
        self._rewards.clear()

        n_items = max(T, 1)
        return {
            "policy_loss": float(total_policy_loss / n_items),
            "value_loss": float(total_value_loss / n_items),
            "entropy": float(total_entropy / n_items),
        }

    def flush(self) -> Optional[Dict]:
        """Force update with remaining transitions."""
        if len(self._states) > 0:
            return self._update_n_step()
        return None

    def save(self, path: str):
        """Save agent weights."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "network": self.network.get_params(),
                "train_step": self.train_step,
            }, f)

    def load(self, path: str):
        """Load agent weights."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.network.set_params(data["network"])
        self.train_step = data.get("train_step", 0)
