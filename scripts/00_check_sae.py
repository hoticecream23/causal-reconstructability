"""Verify the SAE is actually being fed the activations it was trained on.

Run this first, and again for any new model/SAE/layer. It caught two silent bugs already:
left padding corrupting position_ids, and TransformerLens-trained SAEs needing the
residual mean-centred along d_model.

A correctly aligned residual-stream SAE lands near its trained L0 (tens) with explained
variance above ~0.7. If the configured hook point is not the best row in the table, the
hook point is wrong.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse  # noqa: E402

import torch  # noqa: E402

from rnar import data, hooks, model  # noqa: E402
from rnar.cache import tokenize  # noqa: E402
from rnar.config import PRESETS, get_config  # noqa: E402
from rnar.model import get_layers  # noqa: E402

MIN_EV = 0.7
MAX_L0_FRAC = 0.02  # L0 above 2% of d_sae means the encoder is being over-driven


def stats(bundle, resid: torch.Tensor, center: bool) -> tuple[float, float]:
    x = resid.to(torch.float32)
    if center:
        x = x - x.mean(dim=-1, keepdim=True)
    a = bundle.sae.encode(x)
    recon = bundle.sae.decode(a)
    ev = 1 - ((x - recon) ** 2).sum() / ((x - x.mean()) ** 2).sum().clamp_min(1e-9)
    return float((a > 0).float().sum(1).mean()), float(ev)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preset", default="debug", choices=sorted(PRESETS))
    args = p.parse_args()

    cfg = get_config(args.preset)
    cfg.n_prompts = 32
    bundle = model.load(cfg)
    task = data.build(cfg, bundle.tokenizer)
    layers = get_layers(bundle.model)

    print(f"[check] {cfg.model_name} · {cfg.sae_release}/{cfg.sae_id}")
    print(f"[check] d_sae={bundle.d_sae} d_model={bundle.d_model} "
          f"configured hook_layer={cfg.hook_layer} center_resid={cfg.center_resid}\n")

    enc = tokenize(bundle, task.prompts[:16])
    print(f"  {'layer':>5} | {'L0 raw':>9} {'EV raw':>8} | {'L0 centred':>10} {'EV centred':>10}")
    rows = []
    for li in range(len(layers)):
        store: list[torch.Tensor] = []
        with hooks.capture_resid(layers[li], store), torch.no_grad():
            bundle.model(**enc)
        x = store[0].to(cfg.device)
        l0_r, ev_r = stats(bundle, x, center=False)
        l0_c, ev_c = stats(bundle, x, center=True)
        rows.append({"layer": li, "raw": (l0_r, ev_r), "centred": (l0_c, ev_c)})
        mark = " <- configured" if li == cfg.hook_layer else ""
        print(f"  {li:>5} | {l0_r:9.1f} {ev_r:+8.3f} | {l0_c:10.1f} {ev_c:+10.3f}{mark}")

    key = "centred" if cfg.center_resid else "raw"
    best = max(rows, key=lambda r: r[key][1])
    l0, ev = rows[cfg.hook_layer][key]

    print(f"\n[check] configured point: L0 {l0:.1f}  explained variance {ev:+.3f}")
    print(f"[check] best {key} layer is {best['layer']} (EV {best[key][1]:+.3f})")

    ok = True
    if best["layer"] != cfg.hook_layer:
        print(f"  FAIL  hook_layer={cfg.hook_layer} is not the best fit; try {best['layer']}")
        ok = False
    if ev < MIN_EV:
        print(f"  FAIL  explained variance {ev:+.3f} < {MIN_EV}")
        ok = False
    if l0 > MAX_L0_FRAC * bundle.d_sae:
        print(f"  FAIL  L0 {l0:.0f} exceeds {MAX_L0_FRAC:.0%} of d_sae "
              f"({MAX_L0_FRAC * bundle.d_sae:.0f}); encoder is being over-driven")
        ok = False

    # Would the other centring setting be better? That is the bug this script exists for.
    other = "raw" if cfg.center_resid else "centred"
    if rows[cfg.hook_layer][other][1] > ev + 0.05:
        print(f"  FAIL  center_resid={not cfg.center_resid} fits better here "
              f"(EV {rows[cfg.hook_layer][other][1]:+.3f} vs {ev:+.3f}); flip the flag")
        ok = False

    print("\n[check] PASS" if ok else "\n[check] FAILED - fix before running the pipeline")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
