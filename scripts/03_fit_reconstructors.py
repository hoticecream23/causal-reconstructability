"""Fit g_theta for each target on the train split. No model forward passes needed."""

import json

import torch
from _common import parse, path, setup

from rnar import reconstruct
from rnar.cache import Cache


def main() -> None:
    args = parse(__doc__)
    cfg, bundle, _task = setup(args)

    cache = Cache.load(path(cfg, "cache.pt"))
    spec = json.load(open(path(cfg, "targets.json")))
    seeds = {int(k): v for k, v in spec["seeds"].items()}

    fitted = {}
    for t, S in seeds.items():
        X = cache.acts[cache.train_idx][:, S].to(torch.float32)
        y = cache.acts[cache.train_idx, t].to(torch.float32)
        recon = reconstruct.fit(cfg.reconstructor, X, y, cfg.ridge_lambda)

        Xe = cache.acts[cache.eval_idx][:, S].to(torch.float32)
        ye = cache.acts[cache.eval_idx, t].to(torch.float32)
        score = reconstruct.r2(ye, recon.predict(Xe))
        fitted[t] = {"S": S, "recon": recon, "r2_all": score}
        print(f"  feature {t:>6}  |S|={len(S):>2}  R²(held-out) {score:+.3f}")

    out = path(cfg, "reconstructors.pt")
    torch.save(fitted, out)
    print(f"[fit] saved {out}")


if __name__ == "__main__":
    main()
