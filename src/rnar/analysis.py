"""The scatter plot, and the kill criteria checked against it."""

from __future__ import annotations

import json
from pathlib import Path

import torch


def load_results(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def usable(rows: list[dict], min_t: float = 5.0) -> list[dict]:
    """Targets whose ablation moved the metric well clear of noise.

    Gating on an absolute damage threshold does not transfer between tasks: the same 0.1
    cutoff is lenient on a metric of scale 7 and 6x stricter on a metric of scale 1.6.
    `damage_t` is damage over the standard error of the paired per-row difference, which
    is scale-free and is exactly the denominator `R` divides by.
    """
    out = []
    for r in rows:
        t = r.get("damage_t")
        if t is None:  # results from before damage_t existed
            t = abs(r["damage"]) / 0.02
        if t == t and abs(t) >= min_t:
            out.append(r)
    return out


def pearson(x: list[float], y: list[float]) -> float:
    a = torch.tensor(x, dtype=torch.float64)
    b = torch.tensor(y, dtype=torch.float64)
    mask = (a == a) & (b == b)
    a, b = a[mask] - a[mask].mean(), b[mask] - b[mask].mean()
    denom = (a.norm() * b.norm()).clamp_min(1e-12)
    return float((a * b).sum() / denom)


def check_kill_criteria(rows: list[dict]) -> dict:
    good = usable(rows)
    r2s = [r["r2_active"] for r in good]
    Rs = [r["R"] for r in good]
    corr = pearson(r2s, Rs) if len(good) >= 3 else float("nan")

    gap = [r["R"] - r["R_const"] for r in good]
    mean_gap = sum(gap) / len(gap) if gap else float("nan")

    verdict = {
        "n_targets_total": len(rows),
        "n_targets_usable": len(good),
        "corr_R_vs_R2": corr,
        "mean_R_minus_R_const": mean_gap,
        "KILL_distinction_collapsed": bool(corr == corr and corr > 0.9),
        "KILL_const_matches_learned": bool(mean_gap == mean_gap and mean_gap < 0.05),
        "KILL_too_few_usable_targets": len(good) < 10,
    }
    return verdict


def scatter(rows: list[dict], out: Path, title: str = "Causal reconstructability") -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    good = usable(rows)
    x = [r["r2_active"] for r in good]
    y = [r["R"] for r in good]
    yc = [r["R_const"] for r in good]

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.axhline(0, color="0.85", lw=1)
    ax.axvline(0, color="0.85", lw=1)
    ax.plot([0, 1], [0, 1], ls=":", color="0.7", lw=1, label="R = R²")

    ax.scatter(x, yc, s=28, c="0.75", marker="x", label="constant write-back (control)")
    ax.scatter(x, y, s=52, c="#2b6cb0", edgecolor="white", linewidth=0.6, label="learned ĝ(S)")

    ax.set_xlabel("reconstruction quality  R²  (held-out, active rows)")
    ax.set_ylabel("rescue fraction  R")
    ax.set_title(f"{title}\nn={len(good)} targets with measurable damage")
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out
