"""Greedy backward elimination for the minimal sufficient set.

Brute force over subsets is 2^|S|. This is ~|S|^2/2 rescue evaluations per target, and
`Bench` caches the clean and ablated metrics so each evaluation costs one forward pass
rather than three.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import torch

from .cache import Cache, run_multi
from .data import Task
from .reconstruct import fit
from .rescue import _feature_delta, _ratio, active_eval_rows, subtask


class Bench:
    """Fixed per-target state, so the inner loop only pays for the rescue pass.

    A greedy round scores every candidate subset against the same prompts, so they all go
    through `run_multi` in one call and share a single prefix KV cache.
    """

    def __init__(self, bundle, task: Task, cache: Cache, target: int):
        self.bundle = bundle
        self.cache = cache
        self.target = target
        self.rows = active_eval_rows(cache, target)
        self.sub = subtask(task, self.rows)
        self.a = cache.acts[self.rows, target].to(torch.float32)
        self.y_train = cache.acts[cache.train_idx, target].to(torch.float32)

        self.m_clean = float(cache.metric[self.rows].mean())
        self.m_ablated = float(
            run_multi(bundle, self.sub, [_feature_delta(bundle, target, -self.a)])[0].mean()
        )

    def R_for_many(self, subsets: list[list[int]]) -> list[float]:
        cfg = self.bundle.cfg
        fns = []
        for S in subsets:
            X_train = self.cache.acts[self.cache.train_idx][:, S].to(torch.float32)
            recon = fit(cfg.reconstructor, X_train, self.y_train, cfg.ridge_lambda)
            ahat = recon.predict(self.cache.acts[self.rows][:, S].to(torch.float32))
            fns.append(_feature_delta(self.bundle, self.target, ahat - self.a))

        metrics = run_multi(self.bundle, self.sub, fns)
        return [
            _ratio(float(metrics[j].mean()), self.m_ablated, self.m_clean)
            for j in range(len(subsets))
        ]

    def R_for(self, S: list[int]) -> float:
        if not S:
            return float("nan")
        return self.R_for_many([S])[0]


@dataclass
class MinimalSet:
    target: int
    S: list[int]
    R: float
    R_full: float
    seed_size: int
    sufficient: bool

    def to_dict(self) -> dict:
        return asdict(self)


def find(bundle, task: Task, cache: Cache, target: int, seed: list[int], tau: float | None = None,
         verbose: bool = True) -> MinimalSet:
    tau = bundle.cfg.tau if tau is None else tau
    bench = Bench(bundle, task, cache, target)

    S = list(seed)
    R_full = bench.R_for(S)
    if not (R_full >= tau):
        # The full seed set is already insufficient; nothing to minimise.
        return MinimalSet(target, S, R_full, R_full, len(seed), sufficient=False)

    best_R = R_full
    while len(S) > 1:
        trials = [[x for x in S if x != f] for f in S]
        scores = bench.R_for_many(trials)  # one batched pass for the whole round
        scored = list(zip(scores, S, trials))
        # Drop whichever removal leaves rescue highest.
        R_drop, dropped, trial = max(scored, key=lambda t: (t[0] if t[0] == t[0] else -1e9))
        if not (R_drop >= tau):
            break
        S, best_R = trial, R_drop
        if verbose:
            print(f"  target {target}: dropped {dropped}, |S|={len(S)}, R={best_R:.3f}")

    return MinimalSet(target, S, best_R, R_full, len(seed), sufficient=True)
