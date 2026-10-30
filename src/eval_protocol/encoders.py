# -*- coding: utf-8 -*-
"""
eval_protocol/encoders.py
=========================
The single implementation of every representation encoder used in the paper:

* ``IdentityEncoder``  - no-op (raw DRFP)
* ``PCAEncoder``       - unsupervised, linear
* ``StandardAEEncoder``- unsupervised autoencoder
* ``PCLAEEncoder``     - Property Co-Learning AE (supervised by the label y)

All follow ``fit(X, y=None, y_mask=None) -> self`` / ``transform(X)``.

``y_mask`` is the key to a clean leakage decomposition. Only rows with
``y_mask[i] == True`` may contribute to the label (property) loss; *every* row
passed in X contributes to the reconstruction loss. Then

* honest         : fit(X[train], y[train])                      (mask all True)
* transductive_x : fit(X[all],   y_masked, y_mask=is_train)     (X of test seen, y not)
* pretrain_frozen: fit(X[all],   y[all])                        (X and y of test seen)

``lambda_prop`` of ``PCLAEEncoder`` is a **required** argument: it can never
silently default to a stale value.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, TensorDataset


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _check_mask(mask: Optional[np.ndarray], n: int) -> np.ndarray:
    if mask is None:
        return np.ones(n, dtype=bool)
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != (n,):
        raise ValueError(f"y_mask shape {mask.shape} != ({n},)")
    return mask


# ---------------------------------------------------------------------------
class IdentityEncoder(BaseEstimator, TransformerMixin):
    requires_y = False

    def fit(self, X, y=None, y_mask=None):
        return self

    def transform(self, X):
        return np.asarray(X)

    def weights_vector(self) -> np.ndarray:
        return np.zeros(0)


# ---------------------------------------------------------------------------
class PCAEncoder(BaseEstimator, TransformerMixin):
    requires_y = False

    def __init__(self, n_components: int = 64, standardize: bool = True,
                 random_state: int = 0):
        self.n_components = n_components
        self.standardize = standardize
        self.random_state = random_state

    def fit(self, X, y=None, y_mask=None):
        X = np.asarray(X, dtype=np.float64)
        if self.standardize:
            self.mean_ = X.mean(axis=0)
            sd = X.std(axis=0)
            self.scale_ = np.where(sd > 1e-12, sd, 1.0)
        else:
            self.mean_ = np.zeros(X.shape[1])
            self.scale_ = np.ones(X.shape[1])
        k = int(min(self.n_components, X.shape[0], X.shape[1]))
        self.pca_ = PCA(n_components=k, svd_solver="full",
                        random_state=self.random_state)
        self.pca_.fit((X - self.mean_) / self.scale_)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        return self.pca_.transform((X - self.mean_) / self.scale_).astype(np.float32)

    def weights_vector(self) -> np.ndarray:
        return np.concatenate([self.pca_.components_.ravel(), self.mean_, self.scale_])


# ---------------------------------------------------------------------------
class _AEBase(BaseEstimator, TransformerMixin):
    """Shared training loop for the torch autoencoders."""

    requires_y = False
    _supervised = False

    def _build(self, d: int) -> nn.Module:  # pragma: no cover - abstract
        raise NotImplementedError

    def _loss(self, model, xb, yb, mb, y_mean, y_std):  # pragma: no cover
        raise NotImplementedError

    def _fit_input_scaler(self, X):
        if getattr(self, "standardize_input", False):
            self.in_mean_ = X.mean(axis=0)
            sd = X.std(axis=0)
            self.in_scale_ = np.where(sd > 1e-6, sd, 1.0).astype(np.float32)
        else:
            self.in_mean_ = np.zeros(X.shape[1], dtype=np.float32)
            self.in_scale_ = np.ones(X.shape[1], dtype=np.float32)

    def _apply_input_scaler(self, X):
        return ((X - self.in_mean_) / self.in_scale_).astype(np.float32)

    def fit(self, X, y=None, y_mask=None):
        X = np.asarray(X, dtype=np.float32)
        n, d = X.shape
        # input scaling is part of the encoder, so it obeys the same fit-scope
        self._fit_input_scaler(X)
        X = self._apply_input_scaler(X)
        mask = _check_mask(y_mask, n)
        if self._supervised:
            if y is None:
                raise ValueError(f"{type(self).__name__} requires y")
            y = np.asarray(y, dtype=np.float32)
            if mask.sum() < 2:
                raise ValueError("need >= 2 labelled rows for the property loss")
            if not np.isfinite(float(self.lambda_prop)) or self.lambda_prop < 0:
                raise ValueError(f"invalid lambda_prop={self.lambda_prop!r}")
            y_use = np.where(mask, y, 0.0).astype(np.float32)
            y_mean = float(y[mask].mean())
            y_std = float(y[mask].std() + 1e-6)
        else:
            y_use = np.zeros(n, dtype=np.float32)
            y_mean, y_std = 0.0, 1.0

        dev = _resolve_device(self.device)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(self.random_state))
            model = self._build(d).to(dev)
            opt = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=1e-5)
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10, factor=0.5)
            ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y_use),
                               torch.from_numpy(mask.astype(np.float32)))
            gen = torch.Generator().manual_seed(int(self.random_state))
            drop_last = (n % self.batch_size) == 1  # BatchNorm cannot take 1 row
            dl = DataLoader(ds, batch_size=self.batch_size, shuffle=True,
                            generator=gen, drop_last=drop_last)
            for _ in range(self.epochs):
                model.train()
                tot, cnt = 0.0, 0
                for xb, yb, mb in dl:
                    xb, yb, mb = xb.to(dev), yb.to(dev), mb.to(dev)
                    opt.zero_grad()
                    loss = self._loss(model, xb, yb, mb, y_mean, y_std)
                    loss.backward()
                    opt.step()
                    tot += float(loss.item()) * len(xb)
                    cnt += len(xb)
                sched.step(tot / max(cnt, 1))
        model.eval()
        self.model_ = model.cpu()
        self.n_features_in_ = d
        return self

    def transform(self, X):
        X = torch.from_numpy(self._apply_input_scaler(np.asarray(X, dtype=np.float32)))
        self.model_.eval()
        with torch.no_grad():
            return self.model_.encoder(X).numpy()

    def weights_vector(self) -> np.ndarray:
        return np.concatenate([p.detach().cpu().numpy().ravel()
                               for p in self.model_.parameters()])


class _StdAENet(nn.Module):
    def __init__(self, d, latent, hidden):
        super().__init__()
        h1, h2 = hidden
        self.encoder = nn.Sequential(
            nn.Linear(d, h1), nn.BatchNorm1d(h1), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(h1, h2), nn.BatchNorm1d(h2), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(h2, latent),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent, h2), nn.ReLU(),
            nn.Linear(h2, h1), nn.BatchNorm1d(h1), nn.ReLU(),
            nn.Linear(h1, d),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


class _PCLAENet(nn.Module):
    def __init__(self, d, latent, hidden):
        super().__init__()
        h1, h2 = hidden
        self.encoder = nn.Sequential(
            nn.Linear(d, h1), nn.BatchNorm1d(h1), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(h1, h2), nn.BatchNorm1d(h2), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(h2, latent),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent, h2), nn.ReLU(),
            nn.Linear(h2, h1), nn.BatchNorm1d(h1), nn.ReLU(),
            nn.Linear(h1, d),  # logits; sigmoid is folded into BCEWithLogits
        )
        self.predictor = nn.Sequential(
            nn.Linear(latent, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )


class StandardAEEncoder(_AEBase):
    requires_y = False
    _supervised = False

    def __init__(self, latent_dim: int = 64, epochs: int = 100, batch_size: int = 128,
                 lr: float = 1e-3, hidden: Sequence[int] = (512, 256),
                 random_state: int = 0, device: str = "cpu",
                 standardize_input: bool = False):
        self.standardize_input = standardize_input
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.hidden = hidden
        self.random_state = random_state
        self.device = device

    def _build(self, d):
        return _StdAENet(d, self.latent_dim, tuple(self.hidden))

    def _loss(self, model, xb, yb, mb, y_mean, y_std):
        return nn.functional.mse_loss(model(xb), xb)


class PCLAEEncoder(_AEBase):
    """Property Co-Learning AE.  ``lambda_prop`` is mandatory (no default)."""

    requires_y = True
    _supervised = True

    def __init__(self, latent_dim: int, lambda_prop: float, epochs: int = 150,
                 batch_size: int = 128, lr: float = 1e-3, pos_weight: float = 10.0,
                 hidden: Sequence[int] = (512, 256), random_state: int = 0,
                 device: str = "cpu", standardize_input: bool = False):
        self.standardize_input = standardize_input
        self.latent_dim = latent_dim
        self.lambda_prop = lambda_prop
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.pos_weight = pos_weight
        self.hidden = hidden
        self.random_state = random_state
        self.device = device

    def _build(self, d):
        return _PCLAENet(d, self.latent_dim, tuple(self.hidden))

    def _loss(self, model, xb, yb, mb, y_mean, y_std):
        z = model.encoder(xb)
        logits = model.decoder(z)
        pred = model.predictor(z).squeeze(-1)
        w = xb * (self.pos_weight - 1.0) + 1.0
        recon = (nn.functional.binary_cross_entropy_with_logits(
            logits, xb, reduction="none") * w).mean()
        denom = mb.sum().clamp(min=1.0)
        prop = ((((pred * y_std + y_mean) - yb) ** 2) * mb).sum() / denom
        return recon + float(self.lambda_prop) * prop
