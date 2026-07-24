"""Is 'final token only' the reason single-feature ablation barely moves the metric?

The design measures and edits the residual at the final prompt token. But a feature like
"nursing language" fires across the whole bio, so removing it at one position removes a
small fraction of its total contribution -- which would explain damage of <2% and the
useless rescue fractions that follow.

This compares, for the same targets, the damage from ablating a feature at the final token
only against ablating it everywhere it fires. It also reports what share of the feature's
total activation mass sits at the final token, which predicts the gap.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402
from _common import parse, path, setup  # noqa: E402

from rnar import hooks  # noqa: E402
from rnar.cache import Cache, batches, tokenize  # noqa: E402
from rnar.data import metric_from_logits  # noqa: E402
from rnar.rescue import active_eval_rows, subtask  # noqa: E402


@torch.no_grad()
def damage_both_ways(bundle, sub, feat: int) -> tuple[float, float, float, float]:
    """Returns (clean, damage_final_token, damage_all_positions, final-token act share)."""
    direction = bundle.W_dec[feat]
    m_clean = m_final = m_all = 0.0
    mass_final = mass_total = 0.0
    n = 0

    for _sl, prompts, correct, incorrect in batches(sub, bundle.cfg):
        enc = tokenize(bundle, prompts)
        correct = correct.to(bundle.cfg.device)
        incorrect = incorrect.to(bundle.cfg.device)
        mask = enc["attention_mask"]

        logits = bundle.model(**enc).logits
        m_clean += float(metric_from_logits(logits, correct, incorrect).sum())

        # Activations of this feature at every position (masked to real tokens).
        store: list[torch.Tensor] = []

        def grab(_m, _a, output):
            hidden = output[0] if isinstance(output, tuple) else output
            store.append(hidden.detach())

        h = bundle.layer.register_forward_hook(grab)
        bundle.model(**enc)
        h.remove()
        acts = bundle.encode(store[0])[..., feat] * mask  # [b, s]

        mass_final += float(acts[:, -1].sum())
        mass_total += float(acts.sum())

        def final_only(resid):
            return resid.to(torch.float32) - acts[:, -1, None] * direction

        with hooks.edit_resid(bundle.layer, final_only):
            m_final += float(
                metric_from_logits(bundle.model(**enc).logits, correct, incorrect).sum()
            )

        def everywhere(hidden):
            return hidden.to(torch.float32) - acts.unsqueeze(-1) * direction

        with hooks.edit_resid_all(bundle.layer, everywhere):
            m_all += float(
                metric_from_logits(bundle.model(**enc).logits, correct, incorrect).sum()
            )

        n += len(prompts)

    clean = m_clean / n
    share = mass_final / mass_total if mass_total > 0 else float("nan")
    return clean, clean - m_final / n, clean - m_all / n, share


def main() -> None:
    args = parse(__doc__, limit={"type": int, "default": 12, "help": "targets to test"})
    cfg, bundle, task = setup(args)
    cfg.batch_size = min(cfg.batch_size, 8)  # full-sequence SAE acts are memory hungry

    cache = Cache.load(path(cfg, "cache.pt"))
    targets = json.load(open(path(cfg, "targets.json")))["targets"][: args.limit]

    print(f"\n  {'feature':>8} {'n':>5} {'clean':>8} {'dmg_final':>10} {'dmg_all':>9} "
          f"{'ratio':>7} {'final act share':>16}")
    gains = []
    for t in targets:
        rows = active_eval_rows(cache, t)
        if len(rows) < 20:
            continue
        sub = subtask(task, rows)
        clean, d_final, d_all, share = damage_both_ways(bundle, sub, t)
        ratio = d_all / d_final if abs(d_final) > 1e-9 else float("nan")
        gains.append(abs(d_all) / max(abs(d_final), 1e-9))
        print(f"  {t:>8} {len(rows):>5} {clean:>8.3f} {d_final:>10.4f} {d_all:>9.4f} "
              f"{ratio:>7.1f}x {share:>15.1%}")

    if gains:
        med = sorted(gains)[len(gains) // 2]
        print(f"\n[scope] median |damage| multiplier from ablating everywhere: {med:.1f}x")
        print("[scope] if this is large, 'final token only' is what is suppressing damage,")
        print("[scope] and the fix costs the run_multi KV optimisation (edits hit the prefix).")


if __name__ == "__main__":
    main()
