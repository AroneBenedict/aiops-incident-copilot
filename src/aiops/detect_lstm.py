"""LSTM autoencoder detector (optional, needs PyTorch).

The model learns to reconstruct short windows of normal behaviour. A window it cannot
reconstruct well is anomalous. It sees the same seasonal-residual z-scores as the other
detectors, so the comparison isolates the model, not the features.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .features import SeasonalBaseline


class _Autoencoder(nn.Module):
    def __init__(self, n_features: int, hidden: int):
        super().__init__()
        self.encoder = nn.LSTM(n_features, hidden, batch_first=True)
        self.decoder = nn.LSTM(hidden, hidden, batch_first=True)
        self.head = nn.Linear(hidden, n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (state, _) = self.encoder(x)
        repeated = state[-1].unsqueeze(1).expand(-1, x.size(1), -1)
        decoded, _ = self.decoder(repeated)
        return self.head(decoded)


def _windows(z: np.ndarray, length: int) -> np.ndarray:
    """(services, minutes, metrics) -> (services, minutes - length + 1, length, metrics)."""
    view = np.lib.stride_tricks.sliding_window_view(z, length, axis=1)
    return np.moveaxis(view, -1, 2)


class LSTMAutoencoderDetector:
    name = "lstm_autoencoder"

    def __init__(self, seed: int = 0, window: int = 16, hidden: int = 24, epochs: int = 4,
                 train_windows: int = 30_000, clip: float = 12.0):
        self.seed, self.window, self.hidden, self.epochs = seed, window, hidden, epochs
        self.train_windows, self.clip = train_windows, clip
        self.baseline: SeasonalBaseline | None = None
        self.model: _Autoencoder | None = None

    def _prepare(self, metrics: np.ndarray, offset: int = 0) -> np.ndarray:
        z = self.baseline.residual_z(metrics, offset)
        return np.clip(z, -self.clip, self.clip).astype(np.float32)

    def fit(self, clean: np.ndarray) -> LSTMAutoencoderDetector:
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self.baseline = SeasonalBaseline.fit(clean)
        windows = _windows(self._prepare(clean), self.window)
        flat = windows.reshape(-1, self.window, windows.shape[-1])
        if len(flat) > self.train_windows:
            flat = flat[rng.choice(len(flat), self.train_windows, replace=False)]
        data = torch.from_numpy(np.ascontiguousarray(flat))

        self.model = _Autoencoder(data.shape[-1], self.hidden)
        optimiser = torch.optim.Adam(self.model.parameters(), lr=3e-3)
        loss_fn = nn.MSELoss()
        self.model.train()
        for _ in range(self.epochs):
            order = torch.from_numpy(rng.permutation(len(data)))
            for start in range(0, len(data), 256):
                batch = data[order[start : start + 256]]
                optimiser.zero_grad()
                loss = loss_fn(self.model(batch), batch)
                loss.backward()
                optimiser.step()
        self.model.eval()
        return self

    @torch.no_grad()
    def score(self, metrics: np.ndarray, offset: int = 0) -> np.ndarray:
        if self.baseline is None or self.model is None:
            raise RuntimeError("fit() must be called before score()")
        z = self._prepare(metrics, offset)
        services, total, n_features = z.shape
        out = np.zeros((services, total), dtype=np.float64)
        if total < self.window:
            return out
        windows = _windows(z, self.window)
        flat = windows.reshape(-1, self.window, n_features)
        errors = np.empty(len(flat), dtype=np.float64)
        for start in range(0, len(flat), 8192):
            batch = torch.from_numpy(np.ascontiguousarray(flat[start : start + 8192]))
            recent = (self.model(batch) - batch)[:, -5:, :]  # judge the most recent five minutes
            errors[start : start + 8192] = recent.pow(2).mean(dim=(1, 2)).sqrt().numpy()
        out[:, self.window - 1 :] = errors.reshape(services, -1)
        return out
