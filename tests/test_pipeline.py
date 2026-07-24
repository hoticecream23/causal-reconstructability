"""Reconstruction, rescue arithmetic, and split hygiene. No model required."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rnar import reconstruct  # noqa: E402
from rnar.analysis import pearson, usable  # noqa: E402
from rnar.config import Config  # noqa: E402
from rnar.data import split_indices  # noqa: E402
from rnar.rescue import _ratio  # noqa: E402


def test_ridge_recovers_a_linear_target():
    g = torch.Generator().manual_seed(0)
    X = torch.rand(500, 8, generator=g)
    w = torch.tensor([2.0, -1.0, 0.5, 0.0, 0.0, 3.0, 0.0, -0.25])
    y = (X @ w).clamp_min(0.0)

    model = reconstruct.fit_ridge(X[:300], y[:300], lam=1e-3)
    assert reconstruct.r2(y[300:], model.predict(X[300:])) > 0.95


def test_predictions_are_non_negative():
    g = torch.Generator().manual_seed(1)
    X = torch.rand(200, 5, generator=g)
    y = torch.zeros(200)
    y[:50] = 1.0
    model = reconstruct.fit_ridge(X, y)
    assert (model.predict(X) >= 0).all()


def test_r2_of_a_constant_target_is_nan():
    y = torch.ones(50)
    assert reconstruct.r2(y, y) != reconstruct.r2(y, y)  # NaN


def test_rescue_ratio_endpoints():
    # full restoration
    assert _ratio(rescued=1.0, ablated=0.0, clean=1.0) == 1.0
    # no restoration
    assert _ratio(rescued=0.0, ablated=0.0, clean=1.0) == 0.0
    # overshoot is allowed and must not be clipped
    assert _ratio(rescued=1.5, ablated=0.0, clean=1.0) == 1.5


def test_rescue_ratio_is_nan_without_damage():
    r = _ratio(rescued=1.0, ablated=1.0, clean=1.0)
    assert r != r


def _task(prompts):
    from rnar.data import Task

    n = len(prompts)
    return Task(prompts, torch.zeros(n, dtype=torch.long), torch.zeros(n, dtype=torch.long), "t")


def test_train_and_eval_splits_are_disjoint_and_complete():
    cfg = Config(name="test", train_frac=0.6)
    train, ev = split_indices(_task([f"p{i}" for i in range(100)]), cfg)
    assert set(train.tolist()).isdisjoint(ev.tolist())
    assert sorted(train.tolist() + ev.tolist()) == list(range(100))
    assert 55 <= len(train) <= 65


def test_duplicate_prompts_never_straddle_the_split():
    """The leak that pinned every R² at 1.0: same prompt on both sides of the split."""
    cfg = Config(name="test", train_frac=0.6)
    prompts = [f"prompt{i % 8}" for i in range(200)]  # each appears 25 times
    train, ev = split_indices(_task(prompts), cfg)

    train_text = {prompts[i] for i in train.tolist()}
    eval_text = {prompts[i] for i in ev.tolist()}
    assert train_text.isdisjoint(eval_text)


def test_split_rejects_a_task_with_no_holdable_variety():
    cfg = Config(name="test", train_frac=0.6)
    with pytest.raises(RuntimeError, match="not have enough variety"):
        split_indices(_task(["same"] * 50), cfg)


def test_usable_gates_on_significance_not_absolute_size():
    """A small effect measured tightly is usable; a large noisy one is not.

    Gating on |damage| instead threw away 33 of 34 real targets on Bias-in-Bios purely
    because that task's metric has a smaller scale than the synthetic one.
    """
    rows = [
        {"damage": 0.03, "damage_t": 15.0},  # small but t=15
        {"damage": 5.00, "damage_t": 1.2},  # large but indistinguishable from noise
        {"damage": 0.20, "damage_t": float("nan")},
    ]
    keep = usable(rows, min_t=5.0)
    assert [r["damage"] for r in keep] == [0.03]


def test_usable_accepts_negative_damage():
    """Ablation can help the metric; the effect is still real and must not be dropped."""
    assert len(usable([{"damage": -0.1, "damage_t": -12.0}], min_t=5.0)) == 1


def test_pearson_ignores_nan_pairs():
    x = [0.0, 1.0, 2.0, float("nan")]
    y = [0.0, 1.0, 2.0, 5.0]
    assert abs(pearson(x, y) - 1.0) < 1e-6
