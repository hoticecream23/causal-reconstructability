"""Cache SAE activations and the clean metric at the final token of every prompt."""

import json

from _common import parse, path, setup

from rnar import cache as cache_mod


def main() -> None:
    args = parse(__doc__)
    cfg, bundle, task = setup(args)

    # Stamp the run with the SAE it was built from. Downstream analyses index into decoder
    # matrices by feature id, and two different SAEs of the same width would line up
    # silently and produce confident nonsense.
    with open(path(cfg, "run_config.json"), "w") as f:
        json.dump(cfg.to_dict(), f, indent=2)

    cache = cache_mod.build(bundle, task)
    out = path(cfg, "cache.pt")
    cache.save(out)

    active = (cache.acts > 0).float().sum(1)
    print(f"[cache] saved {out}")
    print(f"[cache] acts {tuple(cache.acts.shape)}  mean L0 {active.mean():.1f}")
    print(f"[cache] clean metric  mean {cache.metric.mean():.3f}  "
          f"accuracy(sign>0) {(cache.metric > 0).float().mean():.1%}")
    print(f"[cache] split  train {len(cache.train_idx)}  eval {len(cache.eval_idx)}")


if __name__ == "__main__":
    main()
