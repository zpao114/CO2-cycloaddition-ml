# -*- coding: utf-8 -*-
"""
eval_protocol/models.py
=======================
Single definition of the regressors used by the harness.

``DualBranchANN`` (Conv1d on the first feature block + MLP on the second) is
the architecture of ``utils_benchmark.py`` / ``201_ablation.py`` / ``401`` /
``405``.  ``honest_groupkfold_ablation.py`` historically used a different
network (concatenation -> 64 -> 32 -> 1 MLP with a fixed 60 epochs and no
Conv1d); that one is exposed here under its own name, ``ConcatMLP``, so the two
can never be confused in a table.

``DualBranchRegressor`` / ``ConcatMLPRegressor`` are sklearn-style wrappers
with **nested early stopping**: the validation set used to stop training and to
step the LR scheduler is carved out of the *training fold only*.  If ``groups``
is passed to ``fit`` (e.g. substrates in LOSO), the inner split is a group split
so that the inner validation mimics the outer extrapolation task.  The outer
test fold is never seen.
"""
from __future__ import annotations

import copy
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------
class DualBranchANN(nn.Module):
    """Conv1d-on-first-block + MLP-on-second-block fusion network."""

    def __init__(self, drfp_dim: int, xtb_dim: int, hidden: int = 128):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.bn_drfp = nn.BatchNorm1d(64)
        self.fc_drfp = nn.Sequential(
            nn.Linear(64, hidden), nn.LeakyReLU(0.1), nn.Dropout(0.3),
            nn.Linear(hidden, 64), nn.LeakyReLU(0.1),
        )
        self.fc_xtb = nn.Sequential(
            nn.Linear(xtb_dim, 64), nn.BatchNorm1d(64), nn.LeakyReLU(0.1), nn.Dropout(0.3),
            nn.Linear(64, 32), nn.LeakyReLU(0.1), nn.Dropout(0.2),
        )
        self.fc_out = nn.Sequential(
            nn.Linear(64 + 32, 64), nn.BatchNorm1d(64), nn.LeakyReLU(0.1),
            nn.Linear(64, 1),
        )

    def forward(self, x_first, x_second):
        h = x_first.unsqueeze(1)
        h = torch.relu(self.conv1(h))
        h = torch.relu(self.conv2(h))
        h = self.pool(h).squeeze(-1)
        h = self.fc_drfp(self.bn_drfp(h))
        h2 = self.fc_xtb(x_second)
        return self.fc_out(torch.cat([h, h2], dim=1)).squeeze(-1)


class ConcatMLP(nn.Module):
    """Concatenated-input MLP (64 -> 32 -> 1); the 'honest_groupkfold' variant."""

    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Inner (training-fold-only) validation split
# ---------------------------------------------------------------------------
def inner_validation_split(n: int, groups: Optional[np.ndarray], val_frac: float,
                           seed: int):
    """Return ``(fit_idx, val_idx)`` positions inside the training fold."""
    if groups is not None:
        g = np.asarray(groups)
        if len(np.unique(g)) >= 2:
            gss = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
            fit_idx, val_idx = next(gss.split(np.zeros(n), groups=g))
            if len(fit_idx) >= 2 and len(val_idx) >= 2:
                return fit_idx, val_idx
    ss = ShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
    return next(ss.split(np.zeros(n)))


class _TorchRegressorBase(BaseEstimator, RegressorMixin):
    """Early-stopped torch regressor; subclasses define ``_make_net`` / ``_forward``."""

    def _make_net(self, d: int) -> nn.Module:  # pragma: no cover
        raise NotImplementedError

    def _forward(self, net, xb):  # pragma: no cover
        raise NotImplementedError

    def fit(self, X, y, groups=None):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        n, d = X.shape
        dev = torch.device(self.device)
        fit_idx, val_idx = inner_validation_split(n, groups, self.val_frac,
                                                  int(self.random_state))
        self.inner_val_size_ = int(len(val_idx))

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(self.random_state))
            net = self._make_net(d).to(dev)
            opt = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt, patience=self.lr_patience, factor=0.5)
            Xf = torch.from_numpy(X[fit_idx]); yf = torch.from_numpy(y[fit_idx])
            Xv = torch.from_numpy(X[val_idx]).to(dev)
            yv = torch.from_numpy(y[val_idx]).to(dev)
            gen = torch.Generator().manual_seed(int(self.random_state))
            drop_last = (len(fit_idx) % self.batch_size) == 1
            dl = DataLoader(TensorDataset(Xf, yf), batch_size=self.batch_size,
                            shuffle=True, generator=gen, drop_last=drop_last)
            best, best_state, bad = np.inf, None, 0
            for _ in range(self.max_epochs):
                net.train()
                for xb, yb in dl:
                    xb, yb = xb.to(dev), yb.to(dev)
                    opt.zero_grad()
                    loss = nn.functional.mse_loss(self._forward(net, xb), yb)
                    loss.backward()
                    opt.step()
                net.eval()
                with torch.no_grad():
                    vloss = float(nn.functional.mse_loss(self._forward(net, Xv), yv))
                sched.step(vloss)
                if vloss < best - 1e-9:
                    best, bad = vloss, 0
                    best_state = copy.deepcopy(net.state_dict())
                else:
                    bad += 1
                    if bad >= self.patience:
                        break
            if best_state is not None:
                net.load_state_dict(best_state)
        net.eval()
        self.net_ = net.cpu()
        self.best_val_mse_ = float(best)
        return self

    def predict(self, X):
        X = torch.from_numpy(np.asarray(X, dtype=np.float32))
        self.net_.eval()
        with torch.no_grad():
            return self._forward(self.net_, X).numpy()


class DualBranchRegressor(_TorchRegressorBase):
    """Input ``X = [first_block | second_block]`` with ``n_first`` leading columns.

    The harness sets ``n_first`` automatically (size of the encoded block).
    """

    def __init__(self, n_first: Optional[int] = None, hidden: int = 128,
                 lr: float = 1e-3, weight_decay: float = 1e-4, batch_size: int = 64,
                 max_epochs: int = 300, patience: int = 40, lr_patience: int = 15,
                 val_frac: float = 0.15, random_state: int = 0, device: str = "cpu"):
        self.n_first = n_first
        self.hidden = hidden
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.lr_patience = lr_patience
        self.val_frac = val_frac
        self.random_state = random_state
        self.device = device

    def _make_net(self, d):
        if self.n_first is None or not (0 < self.n_first < d):
            raise ValueError(
                f"DualBranchRegressor needs 0 < n_first < {d}, got {self.n_first!r}")
        return DualBranchANN(self.n_first, d - self.n_first, hidden=self.hidden)

    def _forward(self, net, xb):
        k = self.n_first
        return net(xb[:, :k], xb[:, k:])


class ConcatMLPRegressor(_TorchRegressorBase):
    def __init__(self, lr: float = 1e-3, weight_decay: float = 1e-4, batch_size: int = 64,
                 max_epochs: int = 300, patience: int = 40, lr_patience: int = 15,
                 val_frac: float = 0.15, random_state: int = 0, device: str = "cpu"):
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.lr_patience = lr_patience
        self.val_frac = val_frac
        self.random_state = random_state
        self.device = device

    def _make_net(self, d):
        return ConcatMLP(d)

    def _forward(self, net, xb):
        return net(xb)


def fit_predict_dual(Xd_tr, Xo_tr, y_tr, Xd_te, Xo_te, seed: int = 0,
                     groups=None, **hp) -> np.ndarray:
    """Train ``DualBranchRegressor`` on the training rows (early stopping on an
    inner split of those rows) and return raw (unclipped) test predictions.

    Drop-in replacement for the legacy inline loops that early-stopped on the
    test fold.  ``hp`` are DualBranchRegressor hyper-parameters.
    """
    reg = DualBranchRegressor(n_first=int(Xd_tr.shape[1]), random_state=int(seed), **hp)
    reg.fit(np.hstack([Xd_tr, Xo_tr]).astype(np.float32), np.asarray(y_tr), groups=groups)
    return reg.predict(np.hstack([Xd_te, Xo_te]).astype(np.float32))


# ---------------------------------------------------------------------------
# Tree models - seed is a parameter, never hard-coded
# ---------------------------------------------------------------------------
def make_tree_model(name: str, seed: int = 0):
    """RF / XGB / LGBM with the benchmark hyper-parameters and an explicit seed."""
    if name == "RF":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(n_estimators=200, max_depth=20, min_samples_leaf=2,
                                     n_jobs=-1, random_state=seed)
    if name == "XGB":
        import xgboost as xgb
        return xgb.XGBRegressor(n_estimators=500, max_depth=8, learning_rate=0.05,
                                subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                                random_state=seed, verbosity=0)
    if name == "LGBM":
        import lightgbm as lgb
        return lgb.LGBMRegressor(n_estimators=500, num_leaves=63, learning_rate=0.05,
                                 subsample=0.8, colsample_bytree=0.8,
                                 min_data_in_leaf=10, device="cpu", verbose=-1,
                                 random_state=seed)  # same hyper-parameters as legacy get_tree_model
    raise ValueError(f"unknown tree model {name!r}")
