"""Structured (linear) model of replay gains over the lattice.

Feature of a pull (episode i, change C = (phi, e)):
    effect one-hot, and effect x (atoms that hold at the fork step k_i(phi) of episode i),
    and effect x logged outcome Y_i.
A ridge model x_hat(i, C) = f(i, C)' theta predicts one replay's gain x = Vhat - Y; a
candidate's predicted effect is (1/N) sum_{i in F(phi)} x_hat(i, C). All candidates share
theta, so a pull on one candidate informs every candidate that shares atoms or effect.
"""
from __future__ import annotations

import numpy as np


class LinModel:
    def __init__(self, L, effects):
        self.L = L
        self.effects = list(effects)
        self.A = len(L.atoms)
        self.E = len(self.effects)
        self.d = self.E * (2 + self.A)          # per effect: bias, Y, atoms

    def fork_step(self, combo, i):
        m = 0
        for j in combo:
            m |= 1 << j
        idx = np.nonzero((self.L.masks[i] & m) == m)[0]
        return int(idx[0]) if len(idx) else None

    def feat(self, combo, effect, i):
        k = self.fork_step(combo, i)
        if k is None:
            return None
        e = self.effects.index(effect)
        f = np.zeros(self.d)
        base = e * (2 + self.A)
        f[base] = 1.0
        f[base + 1] = self.L.Y[i]
        mask = int(self.L.masks[i][k])
        for a in range(self.A):
            if mask >> a & 1:
                f[base + 2 + a] = 1.0
        return f

    def fit(self, X, y, lam=1.0):
        X = np.asarray(X)
        y = np.asarray(y)
        P = X.T @ X + lam * np.eye(self.d)
        self.Pinv = np.linalg.inv(P)
        self.theta = self.Pinv @ X.T @ y
        resid = y - X @ self.theta
        self.sigma2 = float(resid @ resid / max(1, len(y) - self.d))
        return self

    def arm_features(self, combo, effect):
        """Mean feature over the arm's affected episodes, scaled by 1/N (so f' theta = predicted Delta)."""
        rows = [self.feat(combo, effect, i) for i in range(self.L.N)]
        rows = [r for r in rows if r is not None]
        if not rows:
            return None
        return np.sum(rows, axis=0) / self.L.N
