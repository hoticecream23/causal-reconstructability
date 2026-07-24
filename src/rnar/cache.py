"""Batched forward passes: tokenization, clean metric, and cached SAE activations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from tqdm import tqdm

from . import hooks
from .config import Config
from .data import Task, metric_from_logits


@dataclass
class Cache:
    acts: torch.Tensor  # [n, d_sae] float16, SAE activations at the final token
    metric: torch.Tensor  # [n] float32, clean signed logit difference
    train_idx: torch.Tensor
    eval_idx: torch.Tensor
    err: torch.Tensor | None = None  # [n, d_model] float16, SAE error term

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.__dict__, path)

    @staticmethod
    def load(path: Path) -> "Cache":
        d = torch.load(path, map_location="cpu", weights_only=True)
        return Cache(**{k: v for k, v in d.items() if k in Cache.__dataclass_fields__})

    def require_err(self) -> torch.Tensor:
        if self.err is None:
            raise RuntimeError(
                "this cache predates the error term; re-run 01_cache_acts.py to add it"
            )
        return self.err


def batches(task: Task, cfg: Config):
    for start in range(0, len(task), cfg.batch_size):
        stop = min(start + cfg.batch_size, len(task))
        yield (
            slice(start, stop),
            task.prompts[start:stop],
            task.correct_ids[start:stop],
            task.incorrect_ids[start:stop],
        )


def tokenize(bundle, prompts: list[str]):
    enc = bundle.tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=bundle.cfg.max_len,
    )
    out = {k: v.to(bundle.cfg.device) for k, v in enc.items()}
    # With left padding, HF still derives position_ids from arange, so every padded row
    # gets its positions shifted by the pad count and its activations are garbage.
    # Rebuild them from the mask so real tokens are always numbered from 0.
    mask = out["attention_mask"]
    out["position_ids"] = (mask.cumsum(-1) - 1).clamp_min(0) * mask
    return out


@torch.no_grad()
def run(bundle, task: Task, edit_fn=None) -> torch.Tensor:
    """Metric for every prompt, optionally under a residual-stream edit.

    `edit_fn` may be a callable applied to every batch, or a function of the batch slice
    returning a per-batch callable (needed when the edit depends on cached activations).
    """
    out = torch.empty(len(task), dtype=torch.float32)
    for sl, prompts, correct, incorrect in batches(task, bundle.cfg):
        enc = tokenize(bundle, prompts)
        fn = _resolve(edit_fn, sl)
        if fn is None:
            logits = bundle.model(**enc).logits
        else:
            with hooks.edit_resid(bundle.layer, fn):
                logits = bundle.model(**enc).logits
        out[sl] = metric_from_logits(
            logits, correct.to(bundle.cfg.device), incorrect.to(bundle.cfg.device)
        ).cpu()
    return out


def per_batch(fn):
    """Mark an edit-fn factory as taking the batch slice."""
    fn.per_batch = True
    return fn


def _resolve(edit_fn, sl):
    if callable(edit_fn) and getattr(edit_fn, "per_batch", False):
        return edit_fn(sl)
    return edit_fn


@torch.no_grad()
def run_multi(bundle, task: Task, edit_fns: list) -> torch.Tensor:
    """Metrics for many residual-stream edits at once: returns [len(edit_fns), len(task)].

    Every edit touches only the final token's residual at the hooked layer, and causal
    attention means no earlier position can depend on it. So the prefix is identical under
    every condition: run it once per batch, keep its KV cache, and evaluate each condition
    as a single-token forward. That turns k full passes over T tokens into one pass over
    T-1 plus k passes over 1 token.

    Numerically equivalent to calling `run` once per edit_fn; `scripts/00b_check_kv.py`
    pins them together against the slow path.
    """
    from contextlib import nullcontext

    k = len(edit_fns)
    out = torch.empty(k, len(task), dtype=torch.float32)

    for sl, prompts, correct, incorrect in batches(task, bundle.cfg):
        enc = tokenize(bundle, prompts)
        ids, mask, pos = enc["input_ids"], enc["attention_mask"], enc["position_ids"]
        correct = correct.to(bundle.cfg.device)
        incorrect = incorrect.to(bundle.cfg.device)
        prefix_len = ids.shape[1] - 1

        if prefix_len < 1:
            # Single-token prompts leave no prefix to amortise; just run them normally.
            for j, fn in enumerate(edit_fns):
                resolved = _resolve(fn, sl)
                ctx = hooks.edit_resid(bundle.layer, resolved) if resolved else nullcontext()
                with ctx:
                    logits = bundle.model(**enc).logits
                out[j, sl] = metric_from_logits(logits, correct, incorrect).cpu()
            continue

        cache = bundle.model(
            input_ids=ids[:, :-1],
            attention_mask=mask[:, :-1],
            position_ids=pos[:, :-1],
            use_cache=True,
        ).past_key_values

        for j, fn in enumerate(edit_fns):
            cache.crop(prefix_len)  # undo the previous condition's appended token
            resolved = _resolve(fn, sl)
            ctx = hooks.edit_resid(bundle.layer, resolved) if resolved else nullcontext()
            with ctx:
                logits = bundle.model(
                    input_ids=ids[:, -1:],
                    attention_mask=mask,
                    position_ids=pos[:, -1:],
                    past_key_values=cache,
                    use_cache=True,
                ).logits
            out[j, sl] = metric_from_logits(logits, correct, incorrect).cpu()

    return out


@torch.no_grad()
def build(bundle, task: Task) -> Cache:
    """One pass over the data collecting the clean metric and SAE activations."""
    cfg = bundle.cfg
    acts = torch.empty(len(task), bundle.d_sae, dtype=torch.float16)
    err = torch.empty(len(task), bundle.d_model, dtype=torch.float16)
    metric = torch.empty(len(task), dtype=torch.float32)

    for sl, prompts, correct, incorrect in tqdm(
        list(batches(task, cfg)), desc="caching", unit="batch"
    ):
        enc = tokenize(bundle, prompts)
        store: list[torch.Tensor] = []
        with hooks.capture_resid(bundle.layer, store):
            logits = bundle.model(**enc).logits
        resid = store[0].to(cfg.device)
        a, e = bundle.split_sae(resid)
        acts[sl] = a.cpu().to(torch.float16)
        err[sl] = e.cpu().to(torch.float16)
        metric[sl] = metric_from_logits(
            logits, correct.to(cfg.device), incorrect.to(cfg.device)
        ).cpu()

    from .data import distinct_ratio, split_indices

    ratio = distinct_ratio(task)
    if ratio < 0.5:
        print(
            f"[cache] WARNING: only {ratio:.0%} of prompts are distinct. The split is "
            "grouped by prompt so this does not leak, but held-out R² will be optimistic "
            "and the eval split may be small."
        )
    train_idx, eval_idx = split_indices(task, cfg)
    return Cache(acts=acts, metric=metric, train_idx=train_idx, eval_idx=eval_idx, err=err)
