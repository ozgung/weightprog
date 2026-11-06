"""Exact, dependency-free statistics for paired task-level comparisons (no scipy).

Every function reproduces the arithmetic of the study's gate scripts exactly (same summation order, same RNG
calls), so a re-analysis returns bit-identical numbers.
"""

from __future__ import annotations

import collections
import math
import random
from typing import Callable, Iterable, Sequence


def sign_test(diffs: Iterable[float]) -> tuple[float, int, int]:
    """One-sided exact sign test that differences are positive: P(X >= #pos), X ~ Bin(#pos + #neg, 1/2).
    Zeros are dropped. Returns (p, n_pos, n_neg); p = 1 when every difference is zero."""
    diffs = list(diffs)
    pos, neg = sum(d > 0 for d in diffs), sum(d < 0 for d in diffs)
    n = pos + neg
    return (sum(math.comb(n, i) for i in range(pos, n + 1)) / 2 ** n if n else 1.0), pos, neg


def binom_two_sided(k: int, n: int) -> float:
    """Two-sided exact binomial test at p = 1/2 (sum of outcomes no more likely than k)."""
    if n == 0:
        return 1.0
    pk = math.comb(n, k)
    return min(1.0, sum(math.comb(n, i) for i in range(n + 1) if math.comb(n, i) <= pk) / 2 ** n)


def holm(pvalues: dict[str, float], m: int | None = None, alpha: float = 0.05) -> dict[str, tuple[float, bool]]:
    """Holm step-down over a family of size m (>= len(pvalues); absent members hold the last slots with p = 1).
    Returns {name: (threshold, passed)}; ties keep the input order (stable sort)."""
    m = max(m or 0, len(pvalues))
    order = sorted(pvalues, key=lambda c: pvalues[c])
    still, out = True, {}
    for i, c in enumerate(order):
        thr = alpha / (m - i)
        ok = still and pvalues[c] < thr
        out[c] = (thr, ok)
        still = ok
    return out


def bootstrap_mean_ci(values: dict[str, float], n: int = 10000, seed: int = 20261104) -> list[float]:
    """Percentile 95 % CI of the mean of `values` (resampling the sorted keys with replacement)."""
    ts = sorted(values)
    rng = random.Random(seed)
    m = sorted(sum(values[rng.choice(ts)] for _ in ts) / len(ts) for _ in range(n))
    return [m[int(0.025 * n)], m[int(0.975 * n) - 1]]


def bootstrap_clustered_ci(values: dict[str, float], cluster_of: dict[str, str] | Callable[[str], str], n: int = 10000,
                           seed: int = 20261105) -> list[float]:
    """Cluster bootstrap: resample clusters (e.g. repositories) with replacement, pool their units, take the mean."""
    get = cluster_of if callable(cluster_of) else cluster_of.__getitem__
    by = collections.defaultdict(list)
    for t in sorted(values):
        by[get(t)].append(values[t])
    cl = sorted(by)
    rng = random.Random(seed)
    mc = []
    for _ in range(n):
        vals = [x for _ in cl for x in by[rng.choice(cl)]]
        mc.append(sum(vals) / len(vals))
    mc.sort()
    return [mc[int(0.025 * n)], mc[int(0.975 * n) - 1]]


def mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs)
