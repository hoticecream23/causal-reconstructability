"""Greedy backward elimination to the minimal sufficient set. Builds the hypergraph."""

import json

from _common import parse, path, setup

from rnar import minimal
from rnar.cache import Cache


def main() -> None:
    args = parse(
        __doc__,
        tau={"type": float, "default": None, "help": "rescue threshold for sufficiency"},
        limit={"type": int, "default": None, "help": "only process the first N targets"},
    )
    cfg, bundle, task = setup(args)

    cache = Cache.load(path(cfg, "cache.pt"))
    spec = json.load(open(path(cfg, "targets.json")))
    seeds = {int(k): v for k, v in spec["seeds"].items()}
    if args.limit:
        seeds = dict(list(seeds.items())[: args.limit])

    # Greedy elimination is ~|S|^2/2 rescue passes per target; worth knowing up front.
    per_target = sum(range(2, cfg.n_candidates + 1))
    print(f"[minimal] {len(seeds)} targets x up to ~{per_target} rescue passes each")

    edges = []
    for t, seed in seeds.items():
        print(f"[minimal] target {t} (seed |S|={len(seed)}, tau={cfg.tau})")
        res = minimal.find(bundle, task, cache, t, seed, tau=cfg.tau)
        edges.append(res.to_dict())
        status = "sufficient" if res.sufficient else "INSUFFICIENT at full seed"
        print(f"  -> |S|={len(res.S)} R={res.R:.3f} ({status})")

    out = path(cfg, "hypergraph.json")
    with open(out, "w") as f:
        json.dump({"tau": cfg.tau, "edges": edges}, f, indent=2)
    print(f"[minimal] saved {out}")

    ok = [e for e in edges if e["sufficient"]]
    if ok:
        avg = sum(len(e["S"]) for e in ok) / len(ok)
        print(f"[minimal] {len(ok)}/{len(edges)} targets reconstructable; mean |S| = {avg:.1f}")


if __name__ == "__main__":
    main()
