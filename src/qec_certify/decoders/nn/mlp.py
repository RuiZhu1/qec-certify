"""Multilayer-perceptron decoder trained by supervised learning on sampled shots.

The network maps a detection-event vector to logits over the ``2^L`` logical classes
(packed observable flips) and is trained with cross-entropy. Cross-entropy is a proper
scoring rule, so with enough data and capacity the softmax converges to the true posterior
``P(l | s)`` and the argmax to the exact ML decoder. That makes exact ML the natural
reference for how far training is from optimal.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from qec_certify.decoders.base import Decoder
from qec_certify.sim import SyndromeData
from qec_certify.utils.bits import pack_bits, unpack_bits

MAX_OBSERVABLES = 10


@dataclass(frozen=True)
class MLPConfig:
    """Architecture and optimisation settings for :class:`MLPDecoder`.

    ``min_steps`` raises the epoch count for small training sets so that every run gets at
    least that many gradient steps; otherwise a small set is under-trained, not just
    data-limited, and the two effects cannot be told apart.
    """

    hidden_sizes: tuple[int, ...] = (64, 64)
    epochs: int = 20
    min_steps: int = 0
    batch_size: int = 1024
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    seed: int = 0


class MLPDecoder(Decoder):
    """Fully connected decoder over detection events.

    Attributes:
        history: Per-epoch ``{"epoch", "train_loss", "val_loss"}`` records from the last
            :meth:`fit` (``val_loss`` is ``None`` without validation data).
    """

    def __init__(self, num_detectors: int, num_observables: int, config: MLPConfig | None = None):
        if not 1 <= num_observables <= MAX_OBSERVABLES:
            raise ValueError(
                f"num_observables must be in [1, {MAX_OBSERVABLES}], got {num_observables}"
            )
        self.num_detectors = num_detectors
        self.num_observables = num_observables
        self.config = config or MLPConfig()
        self.history: list[dict[str, float | int | None]] = []
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.config.seed)
            self.model = self._build_model()

    def _build_model(self) -> nn.Sequential:
        layers: list[nn.Module] = []
        width = self.num_detectors
        for hidden in self.config.hidden_sizes:
            layers += [nn.Linear(width, hidden), nn.ReLU()]
            width = hidden
        layers.append(nn.Linear(width, 2**self.num_observables))
        return nn.Sequential(*layers)

    def fit(self, data: SyndromeData | None = None, val: SyndromeData | None = None) -> MLPDecoder:
        """Train on ``data``. With ``val``, keep the epoch with the lowest validation loss."""
        if data is None:
            raise ValueError("MLPDecoder.fit requires training data")
        if data.shots == 0:
            raise ValueError("training data has no shots")
        cfg = self.config
        features, labels = self._tensors(data)
        steps_per_epoch = -(-len(features) // cfg.batch_size)
        epochs = max(cfg.epochs, -(-cfg.min_steps // steps_per_epoch))
        val_tensors = self._tensors(val) if val is not None else None

        loss_fn = nn.CrossEntropyLoss()
        self.history = []
        best_state, best_val = None, float("inf")
        with torch.random.fork_rng(devices=[]):
            # Distinct stream from the initialisation seed, so shuffling is reproducible too.
            torch.manual_seed(cfg.seed + 1)
            optimizer = torch.optim.Adam(
                self.model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
            )
            for epoch in range(1, epochs + 1):
                self.model.train()
                order = torch.randperm(len(features))
                total = 0.0
                for start in range(0, len(order), cfg.batch_size):
                    idx = order[start : start + cfg.batch_size]
                    optimizer.zero_grad()
                    loss = loss_fn(self.model(features[idx]), labels[idx])
                    loss.backward()
                    optimizer.step()
                    total += loss.item() * len(idx)
                val_loss = self._loss(val_tensors, loss_fn) if val_tensors is not None else None
                self.history.append(
                    {"epoch": epoch, "train_loss": total / len(features), "val_loss": val_loss}
                )
                if val_loss is not None and val_loss < best_val:
                    best_val, best_state = val_loss, copy.deepcopy(self.model.state_dict())
        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.eval()
        return self

    def predict_proba(self, detection_events: np.ndarray) -> np.ndarray:
        """Posterior over logical classes, shape ``(shots, 2^num_observables)``.

        Class ``c`` is the packed observable flips (observable ``j`` is bit ``j``).
        """
        dets = self._check_detection_events(detection_events)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(torch.from_numpy(dets.astype(np.float32)))
            return torch.softmax(logits, dim=1).numpy().astype(np.float64)

    def predict(self, detection_events: np.ndarray) -> np.ndarray:
        classes = np.argmax(self.predict_proba(detection_events), axis=1)
        return unpack_bits(classes, self.num_observables)

    def _tensors(self, data: SyndromeData) -> tuple[torch.Tensor, torch.Tensor]:
        dets = self._check_detection_events(data.detection_events)
        if data.observable_flips.shape[1] != self.num_observables:
            raise ValueError(
                f"expected {self.num_observables} observables, got {data.observable_flips.shape[1]}"
            )
        features = torch.from_numpy(dets.astype(np.float32))
        labels = torch.from_numpy(pack_bits(data.observable_flips))
        return features, labels

    def _loss(self, tensors: tuple[torch.Tensor, torch.Tensor], loss_fn: nn.Module) -> float:
        features, labels = tensors
        self.model.eval()
        with torch.no_grad():
            return loss_fn(self.model(features), labels).item()
