"""Forward hooks for reading and editing the residual stream at the final token.

Everything here operates at sequence position -1. Left padding (set in `model.load`)
guarantees that is the last real token of every prompt in the batch.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Callable

import torch


def _split(output):
    """HF decoder layers return either a bare tensor or a tuple whose [0] is it."""
    if isinstance(output, tuple):
        return output[0], output[1:]
    return output, None


def _rejoin(hidden, rest):
    return hidden if rest is None else (hidden, *rest)


@contextmanager
def edit_resid(layer: torch.nn.Module, fn: Callable[[torch.Tensor], torch.Tensor]):
    """Apply `fn` to the final-token residual stream vector [batch, d_model]."""

    def hook(_module, _args, output):
        hidden, rest = _split(output)
        new_last = fn(hidden[:, -1, :])
        hidden = hidden.clone()
        hidden[:, -1, :] = new_last.to(hidden.dtype)
        return _rejoin(hidden, rest)

    handle = layer.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@contextmanager
def edit_resid_all(layer: torch.nn.Module, fn):
    """Apply `fn` to the whole residual stream [batch, seq, d_model], not just position -1.

    Note this invalidates the assumption `cache.run_multi` depends on: once earlier
    positions change, the prefix KV cache is no longer shared across conditions. Anything
    using this must go through the slow `run` path.
    """

    def hook(_module, _args, output):
        hidden, rest = _split(output)
        return _rejoin(fn(hidden).to(hidden.dtype), rest)

    handle = layer.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@contextmanager
def capture_resid(layer: torch.nn.Module, store: list):
    """Append the final-token residual stream [batch, d_model] to `store`."""

    def hook(_module, _args, output):
        hidden, _ = _split(output)
        store.append(hidden[:, -1, :].detach().to(torch.float32).cpu())

    handle = layer.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@contextmanager
def grad_wrt_features(bundle, store: dict):
    """Make the forward pass differentiable w.r.t. SAE feature activations.

    Rewrites the residual as `x - (a @ W_dec).detach() + a_var @ W_dec` where `a_var`
    is a leaf holding the same values. Numerically an identity; gradients now flow into
    `a_var`, giving the ∂m/∂a term of attribution patching in one backward pass.
    """

    def fn(resid: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            a = bundle.encode(resid)
        a_var = a.clone().requires_grad_(True)
        store["acts"] = a_var
        contrib = a_var @ bundle.W_dec
        return resid.to(torch.float32) - contrib.detach() + contrib

    with edit_resid(bundle.layer, fn):
        yield


def add_along_feature(bundle, feature_idx: int, delta: torch.Tensor):
    """Edit fn adding `delta[b] * W_dec[feature_idx]` to each row's residual."""
    direction = bundle.W_dec[feature_idx]

    def fn(resid: torch.Tensor) -> torch.Tensor:
        d = delta.to(resid.device, torch.float32).unsqueeze(-1)
        return resid.to(torch.float32) + d * direction

    return fn


def add_along_direction(direction: torch.Tensor, delta: torch.Tensor):
    """As above but along an arbitrary (already-scaled) direction."""

    def fn(resid: torch.Tensor) -> torch.Tensor:
        d = delta.to(resid.device, torch.float32).unsqueeze(-1)
        return resid.to(torch.float32) + d * direction.to(resid.device, torch.float32)

    return fn


def add_vectors(vecs: torch.Tensor):
    """Edit fn adding a per-row vector [batch, d_model] to the residual.

    Used to remove or restore the SAE error term, which is a full-rank per-example vector
    rather than a multiple of one decoder direction.
    """

    def fn(resid: torch.Tensor) -> torch.Tensor:
        return resid.to(torch.float32) + vecs.to(resid.device, torch.float32)

    return fn


def subtract_features(bundle, feature_idxs, acts: torch.Tensor):
    """Edit fn removing the contribution of several features at once.

    `acts` is [batch, len(feature_idxs)] holding each feature's clean activation.
    """
    idx = torch.as_tensor(list(feature_idxs), device=bundle.W_dec.device)
    dirs = bundle.W_dec[idx]  # [k, d_model]

    def fn(resid: torch.Tensor) -> torch.Tensor:
        a = acts.to(resid.device, torch.float32)
        return resid.to(torch.float32) - a @ dirs

    return fn
