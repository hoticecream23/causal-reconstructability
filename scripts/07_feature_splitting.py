"""Secondary hypothesis: are reconstructable features the ones that split at wider SAEs?

Splitting score for a narrow feature is the best cosine similarity its decoder direction
achieves against any feature of the wider SAE. Near 1.0 means the wider SAE kept the
feature intact; low means its direction was redistributed across several finer features.

Two modes, and the first gates the second:

  --ladder   task-independent. Walks consecutive widths and reports how much splitting
             actually happens. If max-cos has no spread, there is nothing for R to
             correlate against and the hypothesis is untestable regardless of the task.

  default    correlates the splitting score against R and R² from a rescue.jsonl produced
             at the narrow width.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse  # noqa: E402

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from rnar import analysis  # noqa: E402
from rnar.config import FIGURES, get_config  # noqa: E402

RELEASE = "gpt2-small-res-jb-feature-splitting"
WIDTHS = [768, 1536, 3072, 6144, 12288, 24576, 49152, 98304]


def load_W_dec(width: int, device: str) -> torch.Tensor:
    from sae_lens import SAE

    out = SAE.from_pretrained(RELEASE, f"blocks.8.hook_resid_pre_{width}", device=device)
    sae = out[0] if isinstance(out, tuple) else out
    return sae.W_dec.detach().float()


def max_cosine(narrow: torch.Tensor, wide: torch.Tensor, chunk: int = 2048) -> torch.Tensor:
    """For each narrow row, its best cosine similarity against any wide row."""
    n = F.normalize(narrow, dim=-1)
    w = F.normalize(wide, dim=-1)
    best = []
    for i in range(0, n.shape[0], chunk):
        best.append((n[i : i + chunk] @ w.T).max(dim=1).values)
    return torch.cat(best)


def describe(x: torch.Tensor) -> str:
    q = torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95])
    v = torch.quantile(x, q.to(x.device))
    return "  ".join(f"p{int(p * 100):02d} {val:.3f}" for p, val in zip(q.tolist(), v.tolist()))


def ladder(device: str) -> None:
    print(f"[split] max cosine of each narrow feature against the next width up\n")
    print(f"  {'narrow':>7} -> {'wide':>7}  {'frac<0.9':>9}  quantiles")
    prev = None
    for width in WIDTHS:
        cur = load_W_dec(width, device)
        if prev is not None:
            mc = max_cosine(prev[1], cur)
            frac = float((mc < 0.9).float().mean())
            print(f"  {prev[0]:>7} -> {width:>7}  {frac:>9.1%}  {describe(mc)}")
        prev = (width, cur)
        del cur
        torch.cuda.empty_cache()


def correlate(narrow: int, wide: int, preset: str, device: str) -> None:
    cfg = get_config(preset)
    rows = analysis.usable(analysis.load_results(cfg.run_dir / "rescue.jsonl"))
    if not rows:
        print("[split] no usable rescue results; run 01-04 at the narrow width first")
        return

    # Feature ids only mean the same thing if the rescue run used *this* SAE. Matching
    # widths is not enough -- two releases of equal width have unrelated feature ids.
    import json

    stamp_path = cfg.run_dir / "run_config.json"
    if not stamp_path.exists():
        print(f"[split] {stamp_path} missing; re-run 01_cache_acts.py to stamp the run")
        return
    stamp = json.load(open(stamp_path))
    want_id = f"blocks.8.hook_resid_pre_{narrow}"
    if stamp["sae_release"] != RELEASE or stamp["sae_id"] != want_id:
        print(f"[split] REFUSING to correlate: rescue results came from\n"
              f"          {stamp['sae_release']} / {stamp['sae_id']}\n"
              f"        but the splitting analysis needs\n"
              f"          {RELEASE} / {want_id}\n"
              f"        Feature ids are not comparable across releases even at equal "
              f"width.\n        Run:  python scripts\\01_cache_acts.py --preset split  "
              f"(then 02-04), then retry.")
        return

    mc = max_cosine(load_W_dec(narrow, device), load_W_dec(wide, device)).cpu()
    idx = [r["target"] for r in rows]
    if max(idx) >= len(mc):
        print(f"[split] rescue.jsonl has feature ids up to {max(idx)} but the narrow SAE "
              f"has {len(mc)} features.")
        return

    split_score = [float(mc[i]) for i in idx]
    R = [r["R"] for r in rows]
    R2 = [r["r2_active"] for r in rows]

    print(f"[split] n={len(rows)}  width {narrow} -> {wide}")
    print(f"[split] corr(R,  max-cos) = {analysis.pearson(R, split_score):+.3f}")
    print(f"[split] corr(R², max-cos) = {analysis.pearson(R2, split_score):+.3f}")
    print("[split] hypothesis predicts a NEGATIVE correlation with R "
          "(reconstructable features are the ones that split)")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(split_score, R, s=45, c="#2b6cb0", edgecolor="white", linewidth=0.6)
    ax.set_xlabel(f"max cosine vs. width-{wide} SAE  (low = feature split)")
    ax.set_ylabel("rescue fraction  R")
    ax.set_title(f"Does reconstructability predict splitting?\nwidth {narrow} -> {wide}")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = FIGURES / f"{preset}_splitting.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    print(f"[split] saved {out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ladder", action="store_true", help="task-independent width sweep")
    p.add_argument("--narrow", type=int, default=24576)
    p.add_argument("--wide", type=int, default=49152)
    p.add_argument("--preset", default="debug")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.ladder:
        ladder(device)
    else:
        correlate(args.narrow, args.wide, args.preset, device)


if __name__ == "__main__":
    main()
