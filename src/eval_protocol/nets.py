# -*- coding: utf-8 -*-
"""
eval_protocol/nets.py
=====================
Single definition of the *legacy-compatible* AE networks and the functional
trainers that used to be copy-pasted into 11 scripts
(utils_benchmark, 201, 304, 401, 405, 501, 310, pcl_ae_fair_comparison,
pcl_ae_perfold_retrain, 802, visualization/pcl_ae_visualization).

Why this module exists next to ``encoders.py``
----------------------------------------------
``encoders.py`` holds the sklearn-style ``fit/transform`` encoders used by the
honest harness.  The persisted checkpoints (``pcl_ae_encoder.pt``) and the
per-fold retraining scripts use the ``nn.Module`` + functional-trainer API
below, whose ``state_dict`` layout must stay byte-compatible.  Both families are
now defined exactly once.

Canonical choices (resolved from the drift between the copies)
--------------------------------------------------------------
* property loss is computed on the *original* yield scale
  (``pred * y_std + y_mean`` vs ``y``) - the 304 copy used an un-rescaled loss;
* reconstruction loss is the pos-weighted BCE-with-logits (``pos_weight=10``);
* ``lambda_prop`` has **no silent default** in new code: ``None`` falls back to
  ``config.BEST_LAMBDA_PROP`` and the value actually used is returned in
  ``model.lambda_prop_`` so it can be written to output metadata;
* seeding is optional (``seed=None`` -> legacy behaviour, global RNG).

NOTE (kept for checkpoint compatibility, flagged for the paper): the decoder of
``PropertyCoLearningAE`` ends with ``nn.Sigmoid()`` while the loss applies
``BCEWithLogits`` to it, i.e. a double squashing.  ``encoders.PCLAEEncoder``
(honest harness) uses raw logits.  The two are therefore *not* numerically
identical; do not mix them inside one table.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Networks (state_dict layout identical to every legacy copy)
# ---------------------------------------------------------------------------
class StandardAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, input_dim),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def encode(self, x):
        return self.encoder(x)


class PropertyCoLearningAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, input_dim), nn.Sigmoid(),
        )
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), self.predictor(z).squeeze(-1)

    def encode(self, x):
        return self.encoder(x)


# ---------------------------------------------------------------------------
# Functional trainers
# ---------------------------------------------------------------------------
def _dev(device) -> torch.device:
    if device is None:
        return torch.device("cpu")
    return torch.device(device)


def _loader(tensors, batch_size: int, seed: Optional[int]):
    kw = {}
    if seed is not None:
        kw["generator"] = torch.Generator().manual_seed(int(seed))
    return DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=True, **kw)


def _seed_all(seed: Optional[int]) -> None:
    if seed is not None:
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))


def train_standard_ae_model(X, latent_dim: int, epochs: int = 100, batch_size: int = 128,
                            lr: float = 1e-3, seed: Optional[int] = None,
                            device=None) -> StandardAE:
    """Reconstruction-only AE.  Returns the trained model (eval mode)."""
    _seed_all(seed)
    dev = _dev(device)
    X = np.asarray(X, dtype=np.float32)
    model = StandardAE(X.shape[1], latent_dim).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    crit = nn.MSELoss()
    loader = _loader((torch.from_numpy(X),), batch_size, seed)
    model.train()
    for _ in range(epochs):
        for (xb,) in loader:
            xb = xb.to(dev)
            opt.zero_grad()
            crit(model(xb), xb).backward()
            opt.step()
    model.eval()
    return model


def train_pcl_ae_model(X, y, latent_dim: int, lambda_prop: Optional[float] = None,
                       epochs: int = 150, batch_size: int = 128, lr: float = 1e-3,
                       pos_weight: float = 10.0, seed: Optional[int] = None,
                       device=None, verbose: bool = False) -> PropertyCoLearningAE:
    """Property co-learning AE.  Returns the trained model (eval mode).

    ``y=None`` trains the same network with a zero property term (the
    'standard AE' arm of the fair-comparison scripts).  ``lambda_prop=None``
    resolves to ``config.BEST_LAMBDA_PROP``; the value used is stored in
    ``model.lambda_prop_``.
    """
    if lambda_prop is None:
        try:
            from config import BEST_LAMBDA_PROP as lambda_prop  # type: ignore
        except ImportError:  # pragma: no cover
            from src.config import BEST_LAMBDA_PROP as lambda_prop  # type: ignore
    if lambda_prop is None:
        raise RuntimeError("lambda_prop is None: run 201_ablation.py --stage 2 "
                           "(config.BEST_LAMBDA_PROP is undefined) or pass lambda_prop.")
    lambda_prop = float(lambda_prop)
    _seed_all(seed)
    dev = _dev(device)
    X = np.asarray(X, dtype=np.float32)
    if y is None:
        y_arr = np.zeros(X.shape[0], dtype=np.float32)
        y_mean, y_std = 0.0, 1.0
    else:
        y_arr = np.asarray(y, dtype=np.float32)
        y_mean, y_std = float(y_arr.mean()), float(y_arr.std()) + 1e-6
    Xt = torch.from_numpy(X)
    yt = torch.from_numpy(y_arr).unsqueeze(1)
    model = PropertyCoLearningAE(X.shape[1], latent_dim).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10, factor=0.5)
    bce = nn.BCEWithLogitsLoss(reduction="none")
    mse = nn.MSELoss()
    loader = _loader((Xt, yt), batch_size, seed)
    for ep in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            z = model.encoder(xb)
            recon = model.decoder(z)
            pred = model.predictor(z).squeeze(-1)
            recon_loss = (bce(recon, xb) * (xb * (pos_weight - 1) + 1)).mean()
            prop_loss = mse(pred * y_std + y_mean, yb.squeeze(-1))
            (recon_loss + lambda_prop * prop_loss).backward()
            opt.step()
        sched.step(recon_loss.item())
        if verbose and (ep + 1) % 50 == 0:
            print(f"    epoch {ep + 1}/{epochs}")
    model.eval()
    model.lambda_prop_ = lambda_prop
    return model


@torch.no_grad()
def encode_with_model(model: nn.Module, X, device=None) -> np.ndarray:
    """Encode ``X`` through a frozen AE encoder (float32 numpy out)."""
    model.eval()
    dev = next(model.parameters()).device if device is None else _dev(device)
    z = model.encode(torch.from_numpy(np.asarray(X, dtype=np.float32)).to(dev))
    return z.cpu().numpy().astype(np.float32)


# Latent-returning wrappers (the signature most scripts import).
def train_standard_ae(X, latent_dim: int, epochs: int = 100, batch_size: int = 128,
                      lr: float = 1e-3, seed: Optional[int] = None, device=None) -> np.ndarray:
    m = train_standard_ae_model(X, latent_dim, epochs, batch_size, lr, seed, device)
    return encode_with_model(m, X)


def train_pcl_ae(X, y, latent_dim: int, epochs: int = 150, batch_size: int = 128,
                 lr: float = 1e-3, lambda_prop: Optional[float] = None,
                 pos_weight: float = 10.0, seed: Optional[int] = None,
                 device=None) -> np.ndarray:
    m = train_pcl_ae_model(X, y, latent_dim, lambda_prop, epochs, batch_size, lr,
                           pos_weight, seed, device)
    return encode_with_model(m, X)
