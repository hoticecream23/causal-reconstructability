"""Target selection by indirect effect, via attribution patching to zero ablation.

    IE_i ≈ (a_i^clean − a_i^patch) · ∂m/∂a_i  with  a^patch = 0
         = a_i · ∂m/∂a_i

One backward pass per batch gives the gradient term for all features at once.
"""

from __future__ import annotations

import torch
from tqdm import tqdm

from . import hooks
from .cache import Cache, batches, tokenize
from .data import Task, metric_from_logits


def indirect_effects(bundle, task: Task) -> torch.Tensor:
    """Mean signed IE per feature, [d_sae]."""
    total = torch.zeros(bundle.d_sae, dtype=torch.float32)
    n = 0
    for _sl, prompts, correct, incorrect in tqdm(
        list(batches(task, bundle.cfg)), desc="attribution", unit="batch"
    ):
        enc = tokenize(bundle, prompts)
        store: dict = {}
        with hooks.grad_wrt_features(bundle, store):
            logits = bundle.model(**enc).logits
            m = metric_from_logits(
                logits, correct.to(bundle.cfg.device), incorrect.to(bundle.cfg.device)
            ).sum()
        grad = torch.autograd.grad(m, store["acts"])[0]  # [batch, d_sae]
        total += (store["acts"].detach() * grad).sum(0).float().cpu()
        n += len(prompts)
    return total / n


def pick_targets(ie: torch.Tensor, cache: Cache, cfg) -> list[int]:
    """Top features by |IE|, restricted to ones that actually fire often enough."""
    active_frac = (cache.acts[cache.eval_idx] > 0).float().mean(0)
    eligible = active_frac >= cfg.min_active_frac
    if not eligible.any():
        raise RuntimeError(
            f"no feature fires on >= {cfg.min_active_frac:.1%} of eval prompts; "
            "lower min_active_frac or use more prompts"
        )
    score = ie.abs().clone()
    score[~eligible] = -1.0
    k = min(cfg.n_targets, int(eligible.sum()))
    return score.topk(k).indices.tolist()


def candidate_set(cache: Cache, target: int, cfg) -> list[int]:
    """Seed S: the features most correlated with the target on the *train* split.

    Correlation only ranks candidates; sufficiency is decided later by rescue.
    """
    X = cache.acts[cache.train_idx].to(torch.float32)
    y = X[:, target].clone()
    X[:, target] = 0.0  # never let the target reconstruct itself

    Xc = X - X.mean(0, keepdim=True)
    yc = y - y.mean()
    denom = Xc.norm(dim=0) * yc.norm()
    corr = torch.where(denom > 1e-8, (Xc * yc[:, None]).sum(0) / denom.clamp_min(1e-8),
                       torch.zeros_like(denom))

    k = min(cfg.n_candidates, int((corr.abs() > 0).sum()))
    return corr.abs().topk(k).indices.tolist()
