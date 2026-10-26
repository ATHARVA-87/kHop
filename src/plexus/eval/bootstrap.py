"""Bootstrap confidence intervals over per-query scores (fixed seed: reruns are identical)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class CI:
    mean: float
    low: float
    high: float
    n: int

    def to_dict(self) -> dict:
        return {k: round(v, 4) if isinstance(v, float) else v for k, v in asdict(self).items()}

    def __str__(self) -> str:
        return f"{self.mean:.3f} [{self.low:.3f}, {self.high:.3f}] (n={self.n})"


def bootstrap_ci(
    values: Sequence[float], n_resamples: int = 2000, alpha: float = 0.05, seed: int = 0
) -> CI:
    """Percentile bootstrap CI of the mean."""
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        raise ValueError("no values")
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, x.size, size=(n_resamples, x.size))].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return CI(float(x.mean()), float(lo), float(hi), int(x.size))


@dataclass(frozen=True)
class PairedDiff:
    diff: CI  # mean of (b - a) with CI
    p_not_better: float  # share of resamples where b - a <= 0 (one-sided)

    def to_dict(self) -> dict:
        return {"diff": self.diff.to_dict(), "p_not_better": round(self.p_not_better, 4)}


def paired_bootstrap(
    a: Sequence[float],
    b: Sequence[float],
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> PairedDiff:
    """Is config B better than config A on the SAME queries? Resample queries, not scores."""
    xa, xb = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if xa.shape != xb.shape or xa.size == 0:
        raise ValueError("paired bootstrap needs two equal-length, non-empty score lists")
    d = xb - xa
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, d.size, size=(n_resamples, d.size))].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return PairedDiff(
        CI(float(d.mean()), float(lo), float(hi), int(d.size)), float((means <= 0).mean())
    )
