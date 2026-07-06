"""
Pure NumPy Neural Network Layers
==================================
Dense, Conv1D, and LSTM layers implemented from scratch with
manual forward/backward passes, Xavier initialization, and Adam optimizer.

No PyTorch, no TensorFlow -- just NumPy.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Union


# ---------------------------------------------------------------------------
# Activation functions and their derivatives
# ---------------------------------------------------------------------------

def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def _relu_backward(dout: np.ndarray, cache: np.ndarray) -> np.ndarray:
    return dout * (cache > 0)


def _tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


def _tanh_backward(dout: np.ndarray, cache: np.ndarray) -> np.ndarray:
    return dout * (1 - cache ** 2)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -500, 500)  # prevent overflow
    return 1.0 / (1.0 + np.exp(-x))


def _sigmoid_backward(dout: np.ndarray, cache: np.ndarray) -> np.ndarray:
    return dout * cache * (1 - cache)


def _linear(x: np.ndarray) -> np.ndarray:
    return x


def _linear_backward(dout: np.ndarray, cache: np.ndarray) -> np.ndarray:
    return dout


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


_ACTIVATIONS = {
    "relu": (_relu, _relu_backward),
    "tanh": (_tanh, _tanh_backward),
    "sigmoid": (_sigmoid, _sigmoid_backward),
    "linear": (_linear, _linear_backward),
    None: (_linear, _linear_backward),
}


# ---------------------------------------------------------------------------
# Adam Optimizer
# ---------------------------------------------------------------------------

class Adam:
    """Adam optimizer tracking per-parameter momentum and velocity.

    Uses id(param) as the key so momentum/velocity persist across calls
    for the same parameter array.
    """
    def __init__(self, lr: float = 0.001, beta1: float = 0.9,
                 beta2: float = 0.999, eps: float = 1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m: Dict[int, np.ndarray] = {}
        self.v: Dict[int, np.ndarray] = {}
        self.t = 0

    def update(self, param: np.ndarray, grad: np.ndarray) -> Tuple[int, np.ndarray]:
        pid = id(param)
        # Ensure m/v shapes match current param (handles reshape)
        if pid not in self.m or self.m[pid].shape != param.shape:
            self.m[pid] = np.zeros_like(param)
            self.v[pid] = np.zeros_like(param)
        self.t += 1
        self.m[pid] = self.beta1 * self.m[pid] + (1 - self.beta1) * grad
        self.v[pid] = self.beta2 * self.v[pid] + (1 - self.beta2) * (grad ** 2)
        m_hat = self.m[pid] / (1 - self.beta1 ** self.t)
        v_hat = self.v[pid] / (1 - self.beta2 ** self.t)
        param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        return pid, param


# ---------------------------------------------------------------------------
# Dense (Fully Connected) Layer
# ---------------------------------------------------------------------------

class Dense:
    """Fully connected layer: out = activation(x @ W + b)."""

    def __init__(self, units: int, activation: Optional[str] = "relu",
                 input_dim: Optional[int] = None, name: str = "dense"):
        self.units = units
        self.activation_name = activation
        self.input_dim = input_dim
        self.name = name
        self.W: Optional[np.ndarray] = None
        self.b: Optional[np.ndarray] = None
        self.optimizer = Adam()
        self._built = False
        self._cache: Dict = {}

    def _build(self, input_shape: Tuple[int, ...]):
        if self._built:
            return
        fan_in = input_shape[-1]
        # Xavier uniform initialization
        limit = np.sqrt(6.0 / (fan_in + self.units))
        self.W = np.random.uniform(-limit, limit, (fan_in, self.units))
        self.b = np.zeros((1, self.units))
        self.input_dim = fan_in
        self._built = True

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        self._build(x.shape)
        self._cache["input"] = x
        z = x @ self.W + self.b
        self._cache["z"] = z
        act_fn, _ = _ACTIVATIONS[self.activation_name]
        out = act_fn(z)
        self._cache["activation"] = out
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        _, act_back = _ACTIVATIONS[self.activation_name]
        dz = act_back(dout, self._cache["activation"])
        x = self._cache["input"]
        dW = x.T @ dz / x.shape[0]
        db = np.sum(dz, axis=0, keepdims=True) / x.shape[0]
        dx = dz @ self.W.T
        self.optimizer.update(self.W, dW)
        self.optimizer.update(self.b, db)
        return dx

    @property
    def output_shape(self) -> Tuple[int, ...]:
        return (self.units,)


# ---------------------------------------------------------------------------
# Conv1D Layer
# ---------------------------------------------------------------------------

class Conv1D:
    """1D convolution for time-series data.

    Input shape: (batch, timesteps, channels)
    Output shape: (batch, new_timesteps, filters)
    """
    def __init__(self, filters: int, kernel_size: int, stride: int = 1,
                 padding: str = "same", activation: Optional[str] = "relu",
                 name: str = "conv1d"):
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.activation_name = activation
        self.name = name
        self.W: Optional[np.ndarray] = None
        self.b: Optional[np.ndarray] = None
        self.optimizer = Adam()
        self._built = False
        self._cache: Dict = {}

    def _build(self, input_shape: Tuple[int, ...]):
        if self._built:
            return
        in_channels = input_shape[-1]
        limit = np.sqrt(6.0 / (in_channels * self.kernel_size + self.filters))
        self.W = np.random.uniform(-limit, limit,
                                    (self.kernel_size, in_channels, self.filters))
        self.b = np.zeros((self.filters,))
        self._built = True

    def _pad(self, x: np.ndarray) -> np.ndarray:
        if self.padding == "same":
            pad_w = (self.kernel_size - 1) // 2
            return np.pad(x, ((0, 0), (pad_w, pad_w), (0, 0)), mode="constant")
        return x

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        self._build(x.shape)
        self._cache["input"] = x
        x_padded = self._pad(x)
        B, T, C = x_padded.shape
        # im2col style conv1d
        out_T = (T - self.kernel_size) // self.stride + 1
        cols = np.zeros((B, out_T, self.kernel_size, C))
        for i in range(out_T):
            start = i * self.stride
            cols[:, i, :, :] = x_padded[:, start:start + self.kernel_size, :]
        # cols: (B, out_T, K, C) -> (B, out_T, K*C)
        cols_flat = cols.reshape(B, out_T, -1)
        W_flat = self.W.reshape(-1, self.filters)
        z = cols_flat @ W_flat + self.b  # (B, out_T, F)
        self._cache["cols"] = cols
        self._cache["cols_flat"] = cols_flat
        self._cache["z"] = z
        act_fn, _ = _ACTIVATIONS[self.activation_name]
        out = act_fn(z)
        self._cache["activation"] = out
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        _, act_back = _ACTIVATIONS[self.activation_name]
        dz = act_back(dout, self._cache["activation"])
        B, out_T, F = dz.shape
        cols = self._cache["cols"]
        K, C, _ = self.W.shape
        # Gradient w.r.t W
        dW_flat = self._cache["cols_flat"].transpose(0, 2, 1) @ dz  # (B, K*C, out_T) @ (B, out_T, F) -> (B, K*C, F)
        dW_flat = dW_flat.mean(axis=0)  # (K*C, F)
        dW = dW_flat.reshape(K, C, F)
        db = dz.mean(axis=(0, 1), keepdims=False)  # (F,)
        # Gradient w.r.t input
        W_flat = self.W.reshape(-1, F)
        dcols_flat = dz @ W_flat.T  # (B, out_T, K*C)
        dcols = dcols_flat.reshape(B, out_T, K, C)
        x_padded = self._pad(self._cache["input"])
        dx_padded = np.zeros_like(x_padded)
        for i in range(out_T):
            start = i * self.stride
            dx_padded[:, start:start + K, :] += dcols[:, i, :, :]
        self.optimizer.update(self.W, dW)
        self.optimizer.update(self.b, db)
        if self.padding == "same":
            pad_w = (K - 1) // 2
            return dx_padded[:, pad_w:-pad_w if pad_w > 0 else None, :]
        return dx_padded


# ---------------------------------------------------------------------------
# LSTM Cell
# ---------------------------------------------------------------------------

class LSTM:
    """LSTM layer from scratch.

    Implementation of forget/input/output gates with
    forward and backward passes through time.
    """
    def __init__(self, units: int, return_sequences: bool = False,
                 name: str = "lstm"):
        self.units = units
        self.return_sequences = return_sequences
        self.name = name
        self.W_f: Optional[np.ndarray] = None
        self.U_f: Optional[np.ndarray] = None
        self.b_f: Optional[np.ndarray] = None
        self.W_i: Optional[np.ndarray] = None
        self.U_i: Optional[np.ndarray] = None
        self.b_i: Optional[np.ndarray] = None
        self.W_c: Optional[np.ndarray] = None
        self.U_c: Optional[np.ndarray] = None
        self.b_c: Optional[np.ndarray] = None
        self.W_o: Optional[np.ndarray] = None
        self.U_o: Optional[np.ndarray] = None
        self.b_o: Optional[np.ndarray] = None
        self.optimizer = Adam()
        self._built = False
        self._cache: Dict = {}

    def _build(self, input_shape: Tuple[int, ...]):
        if self._built:
            return
        input_dim = input_shape[-1]
        limit = np.sqrt(6.0 / (input_dim + self.units))
        for gate in ["f", "i", "c", "o"]:
            W = np.random.uniform(-limit, limit, (input_dim, self.units))
            U = np.random.uniform(-limit, limit, (self.units, self.units))
            b = np.zeros((1, self.units))
            setattr(self, f"W_{gate}", W)
            setattr(self, f"U_{gate}", U)
            setattr(self, f"b_{gate}", b)
        self._built = True

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """x: (batch, timesteps, features)"""
        self._build(x.shape)
        B, T, _ = x.shape
        H = self.units

        h = np.zeros((B, H))
        c = np.zeros((B, H))

        # Store all intermediate values for backward pass
        self._cache = {
            "x": x,
            "h_list": [],
            "c_list": [],
            "c_prev_list": [],
            "f_list": [],
            "i_list": [],
            "c_tilde_list": [],
            "o_list": [],
        }

        outputs = []
        for t in range(T):
            xt = x[:, t, :]  # (B, D)
            # Forget gate
            f = _sigmoid(xt @ self.W_f + h @ self.U_f + self.b_f)
            # Input gate
            i = _sigmoid(xt @ self.W_i + h @ self.U_i + self.b_i)
            # Candidate
            c_tilde = _tanh(xt @ self.W_c + h @ self.U_c + self.b_c)
            # Output gate
            o = _sigmoid(xt @ self.W_o + h @ self.U_o + self.b_o)

            c_prev = c.copy()
            c = f * c + i * c_tilde
            h = o * _tanh(c)

            self._cache["f_list"].append(f)
            self._cache["i_list"].append(i)
            self._cache["c_tilde_list"].append(c_tilde)
            self._cache["o_list"].append(o)
            self._cache["c_prev_list"].append(c_prev)
            self._cache["c_list"].append(c.copy())
            self._cache["h_list"].append(h.copy())
            outputs.append(h)

        if self.return_sequences:
            return np.stack(outputs, axis=1)  # (B, T, H)
        return outputs[-1]  # (B, H)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """Backward pass through time."""
        B, T, D = self._cache["x"].shape
        H = self.units
        x = self._cache["x"]

        # Initialize gradients
        dW_f, dU_f, db_f = np.zeros_like(self.W_f), np.zeros_like(self.U_f), np.zeros_like(self.b_f)
        dW_i, dU_i, db_i = np.zeros_like(self.W_i), np.zeros_like(self.U_i), np.zeros_like(self.b_i)
        dW_c, dU_c, db_c = np.zeros_like(self.W_c), np.zeros_like(self.U_c), np.zeros_like(self.b_c)
        dW_o, dU_o, db_o = np.zeros_like(self.W_o), np.zeros_like(self.U_o), np.zeros_like(self.b_o)

        dx = np.zeros_like(x)
        dh_next = np.zeros((B, H))
        dc_next = np.zeros((B, H))

        if not self.return_sequences:
            dout_expanded = np.zeros((B, T, H))
            dout_expanded[:, -1, :] = dout
            dout = dout_expanded

        for t in reversed(range(T)):
            dt = dout[:, t, :] + dh_next  # (B, H)

            o = self._cache["o_list"][t]
            c = self._cache["c_list"][t]
            c_prev = self._cache["c_prev_list"][t]

            # Output gate gradients
            tanh_c = _tanh(c)
            dh = dt
            do = dh * tanh_c
            do_input = _sigmoid_backward(do, o)
            dc = dh * o * _tanh_backward(np.ones_like(c), tanh_c) + dc_next

            # Forget gate
            df = dc * c_prev
            df_input = _sigmoid_backward(df, self._cache["f_list"][t])

            # Input gate
            di = dc * self._cache["c_tilde_list"][t]
            di_input = _sigmoid_backward(di, self._cache["i_list"][t])

            # Candidate
            dc_tilde = dc * self._cache["i_list"][t]
            dc_tilde_input = _tanh_backward(dc_tilde, self._cache["c_tilde_list"][t])

            h_prev = self._cache["h_list"][t - 1] if t > 0 else np.zeros((B, H))
            xt = x[:, t, :]

            dW_f += xt.T @ df_input
            dU_f += h_prev.T @ df_input
            db_f += np.sum(df_input, axis=0, keepdims=True)

            dW_i += xt.T @ di_input
            dU_i += h_prev.T @ di_input
            db_i += np.sum(di_input, axis=0, keepdims=True)

            dW_c += xt.T @ dc_tilde_input
            dU_c += h_prev.T @ dc_tilde_input
            db_c += np.sum(dc_tilde_input, axis=0, keepdims=True)

            dW_o += xt.T @ do_input
            dU_o += h_prev.T @ do_input
            db_o += np.sum(do_input, axis=0, keepdims=True)

            dx[:, t, :] = (df_input @ self.W_f.T +
                           di_input @ self.W_i.T +
                           dc_tilde_input @ self.W_c.T +
                           do_input @ self.W_o.T)

            dh_next = (df_input @ self.U_f.T +
                       di_input @ self.U_i.T +
                       dc_tilde_input @ self.U_c.T +
                       do_input @ self.U_o.T)

            dc_next = dc * self._cache["f_list"][t]

        # Apply gradients
        for (W, dW) in [(self.W_f, dW_f), (self.W_i, dW_i), (self.W_c, dW_c), (self.W_o, dW_o),
                         (self.U_f, dU_f), (self.U_i, dU_i), (self.U_c, dU_c), (self.U_o, dU_o)]:
            self.optimizer.update(W, dW / B)
        for (b, db) in [(self.b_f, db_f), (self.b_i, db_i), (self.b_c, db_c), (self.b_o, db_o)]:
            self.optimizer.update(b, db / B)

        return dx


# ---------------------------------------------------------------------------
# Network Builder
# ---------------------------------------------------------------------------

_LAYER_MAP = {
    "dense": Dense,
    "conv1d": Conv1D,
    "lstm": LSTM,
}


def build_network(config: Union[List[Dict], Dict]) -> List:
    """Build a list of layers from a configuration dict or list.

    Example:
        config = [
            {"type": "dense", "units": 128, "activation": "relu"},
            {"type": "dense", "units": 64, "activation": "relu"},
            {"type": "dense", "units": 3, "activation": "linear"},
        ]
        model = build_network(config)
    """
    if isinstance(config, dict):
        config = config.get("layers", config.get("network", []))

    if not isinstance(config, list):
        raise ValueError("config must be a list of layer dicts or a dict with 'layers' key")

    layers = []
    for i, layer_cfg in enumerate(config):
        cfg = dict(layer_cfg)
        layer_type = cfg.pop("type", "dense").lower()
        if layer_type not in _LAYER_MAP:
            raise ValueError(f"Unknown layer type: {layer_type}. Available: {list(_LAYER_MAP.keys())}")
        layer_cls = _LAYER_MAP[layer_type]
        cfg.setdefault("name", f"{layer_type}_{i}")
        layers.append(layer_cls(**cfg))
    return layers
