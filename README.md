# RNAR: causal reconstructability of SAE features

Can a deleted SAE feature be rebuilt from the other features well enough to restore the
model's *behaviour* and is that different from merely being *predictable* from them?

The full experiment design, controls, and pre-registered kill criteria are in
[docs/DESIGN.md](docs/DESIGN.md). Read that first; it is the spec this code implements.

## Setup

`sae-lens` does not support Python 3.14, so the venv is pinned to 3.11.

```powershell
uv venv --python 3.11
uv pip install -e ".[dev]"
# PyPI ships a CPU-only torch on Windows; pull the CUDA build explicitly or
# everything below runs ~50x slower.
uv pip install "torch==2.11.0" --index-url https://download.pytorch.org/whl/cu128
```

Verify with `.venv\Scripts\python -c "import torch; print(torch.cuda.is_available())"`.

## Run

Everything defaults to the `debug` preset (GPT-2 small, synthetic capitals task, 512
prompts) which runs end to end on a laptop GPU in a few minutes. Use `--preset main` for
Gemma-2-2b + GemmaScope + Bias-in-Bios.

```powershell
.venv\Scripts\python scripts\00_check_sae.py           # GATE: is the SAE fed the right acts?
.venv\Scripts\python scripts\00b_check_kv.py           # GATE: KV fast path == slow path
.venv\Scripts\python scripts\01_cache_acts.py          # SAE acts + clean metric
.venv\Scripts\python scripts\02_select_features.py     # attribution patching -> targets
.venv\Scripts\python scripts\03_fit_reconstructors.py  # ridge g_theta per target
.venv\Scripts\python scripts\04_rescue_eval.py         # rescue + 4 controls
.venv\Scripts\python scripts\05_minimal_sets.py --limit 5   # greedy -> hypergraph
.venv\Scripts\python scripts\06_plot.py                # the scatter + verdict
```

Artifacts land in `data/<preset>/`, figures in `figures/`.

**Always run `00_check_sae.py` first**, and again for any new model/SAE/layer. It exits
non-zero if the SAE is misaligned, and it has already caught two silent bugs that produced
perfectly plausible-looking numbers.

**Step 05 is the bottleneck** - ~`|S|²/2` model passes per target (~435 at the default
`n_candidates=30`). Use `--limit` while iterating. `cache.run_multi` amortises the prompt
prefix across conditions, which took three debug targets from ~12 min to ~115 s; the win
grows with `batch_size`, so do not lower it casually.

### Feature splitting (secondary hypothesis)

```powershell
.venv\Scripts\python scripts\07_feature_splitting.py --ladder     # task-independent sweep
.venv\Scripts\python scripts\01_cache_acts.py --preset split      # then 02-04 --preset split
.venv\Scripts\python scripts\07_feature_splitting.py --preset split
```

The `split` preset exists because **feature ids are not comparable across SAE releases**,
even at identical width. `07` refuses to correlate unless the run was stamped with the
matching release.

## Tests

The hook mechanics are the part that fails silently when wrong, so they are tested against
a fake layer and fake SAE no model downloads, runs in a second.

```powershell
.venv\Scripts\python -m pytest -q
```

## Layout

| path | what |
|---|---|
| `src/rnar/hooks.py` | residual-stream read/edit at the final token; ablation and injection |
| `src/rnar/cache.py` | batched forwards, activation cache, metric under an edit |
| `src/rnar/select.py` | attribution patching, target and candidate-set selection |
| `src/rnar/reconstruct.py` | ridge / MLP `g_theta`, `R²` |
| `src/rnar/rescue.py` | `R`, plus constant / random-direction / ablate-`S` controls |
| `src/rnar/minimal.py` | greedy backward elimination to minimal sufficient sets |
| `src/rnar/analysis.py` | the scatter plot and the kill-criteria check |

## Things that will bite

All four of these were found by running the pipeline, not by reading the code. Each
produced numbers that looked entirely reasonable.

- **Left padding is load-bearing, and needs `position_ids`.** Every hook edits position
  `-1`, and `model.load` sets `padding_side="left"` so that is the last real token for
  every row. But HF derives `position_ids` from `arange` regardless of the mask, so padded
  rows silently get shifted positions - L0 6613 instead of 625. `cache.tokenize` rebuilds
  `position_ids` from the attention mask; do not drop that.
- **SAEs trained on TransformerLens need the residual mean-centred** along `d_model` (TL's
  `center_writing_weights`); raw HF activations are not centred and the offset grows with
  depth. Uncentred at layer 7: L0 625, explained variance −2.07. Centred: L0 74, EV +0.91.
  That is the `center_resid` flag, and `00_check_sae.py` is what tells you which way to set
  it. Only encoding is affected - decoder rows are mean-zero to within 4% of their norm and
  LayerNorm removes that component anyway, so injection still writes into the raw stream.
- **Split by prompt text, not row index.** Any repeated prompt otherwise lands on both
  sides of the split and every R² pins at 1.000.
- **Feature ids mean nothing across SAE releases.** Two releases of the same width have
  unrelated ids, so indexing one run's results into the other's decoder lines up silently
  and yields confident nonsense. Runs are stamped in `run_config.json`; keep that check.
- **Never judge damage by absolute size.** Effects of 0.03 on a metric of scale 1.6 are
  `t` = 15 across 800 paired rows. An absolute threshold does not transfer between tasks
  and once discarded 33 of 34 real targets. Gate on `damage_t`.
- **Bias-in-Bios is ~86/14 professor:nurse.** Unbalanced, a constant answer scores 86% and
  the metric mostly measures the class prior. `data.py` samples the classes evenly.
- **Its `profession` column is a bare int with no label names**, so the id mapping comes
  from the alphabetical profession list in `data.py` (professor = 21, nurse = 13, verified
  by inspecting bios). Re-check it if the dataset is ever revised.
- **`hook_layer` is not the SAE's layer number.** SAEs trained on `blocks.N.hook_resid_pre`
  see the output of layer `N-1`. The debug preset uses `hook_layer=7` for
  `blocks.8.hook_resid_pre`; GemmaScope `resid_post` SAEs use the layer number directly.
- **Ablation subtracts from the real residual stream**, not from the SAE reconstruction, so
  the error term survives. Replacing `x` with `x̂` would confound every result.
- **Rescue is only measured on rows where the target fires.** Elsewhere ablation is a no-op
  and `R` is 0/0.
- **Bias-in-Bios profession label ids** fall back to hard-coded values if the dataset ships
  no `ClassLabel` names. The script warns; verify before trusting a run.
