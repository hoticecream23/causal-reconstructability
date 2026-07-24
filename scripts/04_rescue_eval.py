"""Ablate-and-rescue every target, with all four controls. Produces rescue.jsonl."""

import json

import torch
from _common import parse, path, setup

from rnar import rescue
from rnar.cache import Cache


def main() -> None:
    args = parse(__doc__)
    cfg, bundle, task = setup(args)

    cache = Cache.load(path(cfg, "cache.pt"))
    fitted = torch.load(path(cfg, "reconstructors.pt"), weights_only=False)

    out = path(cfg, "rescue.jsonl")
    with open(out, "w") as f:
        for t, entry in fitted.items():
            res = rescue.evaluate(bundle, task, cache, t, entry["S"], entry["recon"])
            f.write(json.dumps(res.to_dict()) + "\n")
            f.flush()
            print(
                f"  feature {t:>6}  n={res.n_active:>4}  damage {res.damage:+.3f} "
                f"(t={res.damage_t:+6.1f})  R² {res.r2_active:+.2f}  R {res.R:+.2f}  "
                f"[const {res.R_const:+.2f}  rand {res.R_randdir:+.2f}]"
            )

    print(f"[rescue] saved {out}")


if __name__ == "__main__":
    main()
