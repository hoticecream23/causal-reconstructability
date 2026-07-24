"""Ablate-and-rescue evaluation, with the four controls the design doc requires.

Rescue is only meaningful where the target actually fires: on rows with `a_i == 0`
ablation is a no-op and `R` would be 0/0. Every condition here is therefore evaluated on
the *active eval rows* only.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import torch

from . import hooks
from .cache import Cache, per_batch, run_multi
from .data import Task
from .reconstruct import r2


@dataclass
class RescueResult:
    target: int
    S: list[int]
    n_active: int
    r2_all: float
    r2_active: float
    m_clean: float
    m_ablated: float
    m_rescued: float
    m_const: float
    m_randdir: float
    m_ablate_S: float
    damage: float
    damage_se: float  # standard error of the paired per-row damage
    damage_t: float  # damage / damage_se; how far the effect sits above noise
    R: float
    R_const: float
    R_randdir: float

    def to_dict(self) -> dict:
        return asdict(self)


def active_eval_rows(cache: Cache, target: int) -> torch.Tensor:
    idx = cache.eval_idx
    return idx[cache.acts[idx, target] > 0]


def subtask(task: Task, rows: torch.Tensor) -> Task:
    r = rows.tolist()
    return Task(
        prompts=[task.prompts[i] for i in r],
        correct_ids=task.correct_ids[rows],
        incorrect_ids=task.incorrect_ids[rows],
        name=f"{task.name}[{len(r)}]",
    )


def _ratio(rescued: float, ablated: float, clean: float) -> float:
    denom = clean - ablated
    if abs(denom) < 1e-6:
        return float("nan")
    return (rescued - ablated) / denom


def _feature_delta(bundle, feat: int, delta: torch.Tensor):
    @per_batch
    def factory(sl):
        return hooks.add_along_feature(bundle, feat, delta[sl])

    return factory


def _random_direction_fn(bundle, feat: int, a: torch.Tensor, ahat: torch.Tensor, other: int):
    """Ablate along W_dec[feat], write the same predicted value along a norm-matched
    unrelated feature direction. Isolates 'restored the right thing' from 'restored norm'."""
    d_i = bundle.W_dec[feat]
    u = bundle.W_dec[other]
    u = u * (d_i.norm() / u.norm().clamp_min(1e-8))

    @per_batch
    def factory(sl):
        def fn(resid):
            x = resid.to(torch.float32)
            x = x - a[sl].to(x.device)[:, None] * d_i
            return x + ahat[sl].to(x.device)[:, None] * u

        return fn

    return factory


def evaluate(bundle, task: Task, cache: Cache, target: int, S: list[int], recon) -> RescueResult:
    cfg = bundle.cfg
    rows = active_eval_rows(cache, target)
    sub = subtask(task, rows)

    a = cache.acts[rows, target].to(torch.float32)
    X_eval = cache.acts[rows][:, S].to(torch.float32)
    ahat = recon.predict(X_eval)

    # R^2 on all held-out rows (honest prediction task) and on active rows only.
    all_idx = cache.eval_idx
    r2_all = r2(
        cache.acts[all_idx, target].to(torch.float32),
        recon.predict(cache.acts[all_idx][:, S].to(torch.float32)),
    )
    r2_active = r2(a, ahat)

    # Constant write-back: the mean active activation on train, no dependence on S.
    tr = cache.train_idx
    tr_active = cache.acts[tr, target].to(torch.float32)
    const_val = float(tr_active[tr_active > 0].mean()) if (tr_active > 0).any() else 0.0
    const = torch.full_like(a, const_val)

    g = torch.Generator().manual_seed(cfg.seed + target)
    other = int(torch.randint(bundle.d_sae, (1,), generator=g))
    while other == target:
        other = int(torch.randint(bundle.d_sae, (1,), generator=g))

    acts_S = cache.acts[rows][:, S].to(torch.float32)

    @per_batch
    def ablate_S(sl):
        return hooks.subtract_features(bundle, S, acts_S[sl])

    # All five conditions share the same prompts, so they share one prefix KV cache.
    metrics = run_multi(
        bundle,
        sub,
        [
            _feature_delta(bundle, target, -a),
            _feature_delta(bundle, target, ahat - a),
            _feature_delta(bundle, target, const - a),
            _random_direction_fn(bundle, target, a, ahat, other),
            ablate_S,
        ],
    )
    clean_per_row = cache.metric[rows]
    m_clean = float(clean_per_row.mean())
    m_ablated, m_rescued, m_const, m_randdir, m_ablate_S = (float(m.mean()) for m in metrics)

    # Damage is a paired difference over the same rows, so its noise is the SE of that
    # difference -- not of the metric itself. `R` divides by this quantity, so when it is
    # not comfortably above noise every downstream number is meaningless.
    damage_per_row = clean_per_row - metrics[0]
    n = max(len(damage_per_row), 2)
    damage_se = float(damage_per_row.std(unbiased=True) / (n**0.5))
    damage = m_clean - m_ablated

    return RescueResult(
        target=target,
        S=list(S),
        n_active=len(rows),
        r2_all=r2_all,
        r2_active=r2_active,
        m_clean=m_clean,
        m_ablated=m_ablated,
        m_rescued=m_rescued,
        m_const=m_const,
        m_randdir=m_randdir,
        m_ablate_S=m_ablate_S,
        damage=damage,
        damage_se=damage_se,
        damage_t=damage / damage_se if damage_se > 1e-12 else float("nan"),
        R=_ratio(m_rescued, m_ablated, m_clean),
        R_const=_ratio(m_const, m_ablated, m_clean),
        R_randdir=_ratio(m_randdir, m_ablated, m_clean),
    )


# The minimality search uses `minimal.Bench` instead, which caches the clean and ablated
# metrics across a whole greedy round rather than recomputing them per candidate.
