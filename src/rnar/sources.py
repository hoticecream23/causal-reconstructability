"""Where does a feature's recoverable information live: the dictionary, or its residual?

Cui, Shen & Yang (arXiv 2606.18322) show that behaviour suppressed by an SAE feature
intervention can be recovered, and attribute that recovery to the SAE reconstruction
residual -- the part the dictionary never explained. That predicts our core effect should
be weak: if the redundancy lives outside the dictionary, other features should not be able
to stand in for an ablated one.

This module tests the information side of that claim directly. For each target we fit the
same reconstructor from three input sources and compare both prediction (R²) and causal
restoration (R):

    features   other SAE latents in S          -- dictionary-internal
    error      the SAE error term, PCA'd       -- dictionary-external
    both       concatenation                   -- ceiling

To keep it a fair fight the error term is projected to exactly |S| components, so all three
see comparable input dimension (`both` gets 2|S| and is a ceiling, not a rival).

Scope note: this is not a replication of 2606.18322. They optimise an adversarial,
per-input, encoder-orthogonal perturbation; we ask where a feature's *value* is recoverable
from. Related question, different construct -- do not claim to have refuted them.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

from . import hooks
from .cache import Cache, per_batch, run_multi
from .data import Task
from .reconstruct import fit, r2
from .rescue import _feature_delta, _ratio, active_eval_rows, subtask


@dataclass
class SourceResult:
    target: int
    n_active: int
    k: int
    r2_feat: float
    r2_err: float
    r2_both: float
    R_feat: float
    R_err: float
    R_both: float
    # Metric shift from deleting the whole error term, expressed in units of this target's
    # single-feature damage. NOT a rescue fraction -- deleting the error term is a
    # full-rank perturbation, so values far above 1 just mean the residual carries far more
    # causal weight than one feature does.
    R_dict_only: float
    m_clean: float
    m_ablated: float
    damage: float
    damage_t: float

    def to_dict(self) -> dict:
        return asdict(self)


def pca_basis(x: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-k principal directions of `x`, fitted on train rows only."""
    mu = x.mean(0, keepdim=True)
    _u, _s, v = torch.pca_lowrank(x - mu, q=min(k + 8, min(x.shape) - 1))
    return mu, v[:, :k]


def evaluate(bundle, task: Task, cache: Cache, target: int, S: list[int]) -> SourceResult:
    cfg = bundle.cfg
    err_all = cache.require_err()
    rows = active_eval_rows(cache, target)
    sub = subtask(task, rows)
    tr = cache.train_idx
    k = len(S)

    y_tr = cache.acts[tr, target].to(torch.float32)
    y_ev = cache.acts[rows, target].to(torch.float32)

    feat_tr = cache.acts[tr][:, S].to(torch.float32)
    feat_ev = cache.acts[rows][:, S].to(torch.float32)

    mu, basis = pca_basis(err_all[tr].to(torch.float32), k)
    err_tr = (err_all[tr].to(torch.float32) - mu) @ basis
    err_ev = (err_all[rows].to(torch.float32) - mu) @ basis

    both_tr = torch.cat([feat_tr, err_tr], dim=1)
    both_ev = torch.cat([feat_ev, err_ev], dim=1)

    preds = {}
    for name, (X_tr, X_ev) in {
        "feat": (feat_tr, feat_ev),
        "err": (err_tr, err_ev),
        "both": (both_tr, both_ev),
    }.items():
        model = fit(cfg.reconstructor, X_tr, y_tr, cfg.ridge_lambda)
        preds[name] = model.predict(X_ev)

    err_rows = err_all[rows].to(torch.float32)

    @per_batch
    def drop_error(sl):
        return hooks.add_vectors(-err_rows[sl])

    metrics = run_multi(
        bundle,
        sub,
        [
            _feature_delta(bundle, target, -y_ev),
            _feature_delta(bundle, target, preds["feat"] - y_ev),
            _feature_delta(bundle, target, preds["err"] - y_ev),
            _feature_delta(bundle, target, preds["both"] - y_ev),
            drop_error,
        ],
    )

    clean_per_row = cache.metric[rows]
    m_clean = float(clean_per_row.mean())
    m_abl, m_feat, m_err, m_both, m_dict = (float(m.mean()) for m in metrics)

    damage_per_row = clean_per_row - metrics[0]
    n = max(len(damage_per_row), 2)
    se = float(damage_per_row.std(unbiased=True) / (n**0.5))
    damage = m_clean - m_abl

    return SourceResult(
        target=target,
        n_active=len(rows),
        k=k,
        r2_feat=r2(y_ev, preds["feat"]),
        r2_err=r2(y_ev, preds["err"]),
        r2_both=r2(y_ev, preds["both"]),
        R_feat=_ratio(m_feat, m_abl, m_clean),
        R_err=_ratio(m_err, m_abl, m_clean),
        R_both=_ratio(m_both, m_abl, m_clean),
        R_dict_only=_ratio(m_dict, m_abl, m_clean),
        m_clean=m_clean,
        m_ablated=m_abl,
        damage=damage,
        damage_t=damage / se if se > 1e-12 else float("nan"),
    )
