"""Pin the KV-cache fast path to the slow path, and measure the speedup.

`run_multi` claims that evaluating an edit as a single-token forward on a cached prefix is
identical to re-running the whole prompt. That is only true because the edit touches the
final position and attention is causal. If either assumption breaks, results stay
plausible-looking and are wrong -- so this asserts equality on real activations.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse  # noqa: E402

import torch  # noqa: E402

from rnar import data, hooks, model  # noqa: E402
from rnar.cache import Cache, per_batch, run, run_multi  # noqa: E402
from rnar.config import PRESETS, get_config  # noqa: E402
from rnar.rescue import _feature_delta, active_eval_rows, subtask  # noqa: E402

TOL = 2e-3  # bf16/fp32 kernels differ slightly between batched and single-token paths


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preset", default="debug", choices=sorted(PRESETS))
    p.add_argument("--k", type=int, default=8, help="number of conditions to compare")
    p.add_argument("--batch", type=int, default=None, help="override batch_size")
    args = p.parse_args()

    cfg = get_config(args.preset)
    if args.batch:
        cfg.batch_size = args.batch
    bundle = model.load(cfg)
    task = data.build(cfg, bundle.tokenizer)
    cache = Cache.load(cfg.run_dir / "cache.pt")

    target = int(cache.acts[cache.eval_idx].sum(0).argmax())
    rows = active_eval_rows(cache, target)
    sub = subtask(task, rows)
    a = cache.acts[rows, target].to(torch.float32)
    print(f"[kv] target {target}, {len(rows)} active rows, {args.k} conditions")

    # A spread of conditions: full ablation through to full restoration, plus clean.
    fns = [None] + [
        _feature_delta(bundle, target, (scale - 1.0) * a)
        for scale in torch.linspace(0.0, 1.0, args.k)
    ]

    t0 = time.perf_counter()
    slow = torch.stack([run(bundle, sub, fn) for fn in fns])
    t_slow = time.perf_counter() - t0

    t0 = time.perf_counter()
    fast = run_multi(bundle, sub, fns)
    t_fast = time.perf_counter() - t0

    diff = (slow - fast).abs()
    print(f"[kv] slow {t_slow:6.2f}s   fast {t_fast:6.2f}s   speedup {t_slow / t_fast:5.2f}x")
    print(f"[kv] max abs diff {diff.max():.2e}   mean {diff.mean():.2e}")

    # A wrong prefix cache shows up as a constant offset, so check the metric spread too.
    print(f"[kv] metric range slow [{slow.min():+.3f}, {slow.max():+.3f}]  "
          f"fast [{fast.min():+.3f}, {fast.max():+.3f}]")

    if diff.max() > TOL:
        worst = int(diff.max(dim=1).values.argmax())
        print(f"\n[kv] FAILED: condition {worst} differs by {diff[worst].max():.2e} > {TOL}")
        sys.exit(1)

    print("\n[kv] PASS")


if __name__ == "__main__":
    main()
