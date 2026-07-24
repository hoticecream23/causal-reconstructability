"""Is an ablated feature recoverable from the dictionary, or only from its residual?

This is the experiment the novelty review pointed at: arXiv 2606.18322 attributes
post-intervention behavioural recovery to the SAE reconstruction residual, which predicts
that other features should NOT be able to stand in for an ablated one. Here we fit the same
reconstructor from other features, from the error term, and from both, and compare.

Read `usable` rows only -- targets whose ablation did not clear the noise floor say nothing
either way.
"""

import json

from _common import parse, path, setup

from rnar import analysis, sources
from rnar.cache import Cache


def main() -> None:
    args = parse(__doc__, limit={"type": int, "default": None, "help": "first N targets"})
    cfg, bundle, task = setup(args)

    cache = Cache.load(path(cfg, "cache.pt"))
    spec = json.load(open(path(cfg, "targets.json")))
    seeds = {int(k): v for k, v in spec["seeds"].items()}
    if args.limit:
        seeds = dict(list(seeds.items())[: args.limit])

    out = path(cfg, "sources.jsonl")
    rows = []
    with open(out, "w") as f:
        for t, S in seeds.items():
            res = sources.evaluate(bundle, task, cache, t, S)
            f.write(json.dumps(res.to_dict()) + "\n")
            f.flush()
            rows.append(res.to_dict())
            print(
                f"  feature {t:>6}  t={res.damage_t:+6.1f}  "
                f"R²[feat {res.r2_feat:+.2f} err {res.r2_err:+.2f} both {res.r2_both:+.2f}]  "
                f"R[feat {res.R_feat:+.2f} err {res.R_err:+.2f} both {res.R_both:+.2f}]"
            )

    print(f"\n[sources] saved {out}")

    good = analysis.usable(rows)
    print(f"[sources] {len(good)}/{len(rows)} targets cleared the damage noise floor")
    if not good:
        print("[sources] nothing interpretable: ablation never moved the metric. "
              "Sharpen the task before reading anything into these numbers.")
        return

    def mean(key):
        vals = [r[key] for r in good if r[key] == r[key]]
        return sum(vals) / len(vals) if vals else float("nan")

    print(f"\n  {'source':<8} {'mean R²':>9} {'mean R':>9}")
    for src, lbl in [("feat", "features"), ("err", "error"), ("both", "both")]:
        print(f"  {lbl:<8} {mean('r2_' + src):>9.3f} {mean('R_' + src):>9.3f}")
    print(f"\n  deleting the error term moves the metric {abs(mean('R_dict_only')):.1f}x "
          f"as far as ablating one target feature does")

    r2_gap = mean("r2_feat") - mean("r2_err")
    print(f"\n[sources] R² advantage of dictionary over residual: {r2_gap:+.3f}")
    if r2_gap > 0.1:
        print("  -> feature information is dictionary-internal, which cuts AGAINST "
              "2606.18322's residual attribution")
    elif r2_gap < -0.1:
        print("  -> feature information is carried by the residual, CONSISTENT with "
              "2606.18322")
    else:
        print("  -> neither source dominates; report as inconclusive, not as support")


if __name__ == "__main__":
    main()
