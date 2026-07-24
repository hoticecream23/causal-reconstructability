"""Rank features by indirect effect and pick the targets, plus each target's seed set S."""

import json

import torch
from _common import parse, path, setup

from rnar import select
from rnar.cache import Cache


def main() -> None:
    args = parse(__doc__)
    cfg, bundle, task = setup(args)

    cache = Cache.load(path(cfg, "cache.pt"))
    ie = select.indirect_effects(bundle, task)
    torch.save(ie, path(cfg, "indirect_effects.pt"))

    targets = select.pick_targets(ie, cache, cfg)
    seeds = {t: select.candidate_set(cache, t, cfg) for t in targets}

    out = path(cfg, "targets.json")
    with open(out, "w") as f:
        json.dump({"targets": targets, "seeds": {str(k): v for k, v in seeds.items()}}, f, indent=2)

    print(f"[select] saved {out}")
    print(f"[select] {len(targets)} targets; top by |IE|:")
    for t in targets[:10]:
        frac = (cache.acts[cache.eval_idx, t] > 0).float().mean()
        print(f"  feature {t:>6}  IE {ie[t]:+.4f}  fires on {frac:.1%} of eval  |S_seed|={len(seeds[t])}")


if __name__ == "__main__":
    main()
