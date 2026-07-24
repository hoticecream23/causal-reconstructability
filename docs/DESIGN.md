# Causal Reconstructability of SAE Features

**Question.** If we delete an SAE feature `f_i` from a model, can we rebuild it well enough
from the *other* features that the model's behaviour is restored — and is that the same thing
as being able to *predict* `f_i` from the other features?

The claim under test is that these two things come apart. Prediction is representational
redundancy. Restoration is functional interchangeability. Existing work (sparse probing,
feature-circuit correlation) measures the first and quietly assumes the second.

## The plot the whole project is built around

For each target feature `f_i`:

- **x-axis — reconstruction quality `R²`**: how well `ĝ_θ(S)` predicts `f_i`'s activation on held-out prompts.
- **y-axis — rescue fraction `R`**: how much of `f_i`'s causal contribution is restored when
  `ĝ_θ(S)` is injected into the residual stream after `f_i` is ablated.

```
R = (m_rescued − m_ablated) / (m_clean − m_ablated)
```

where `m` is a scalar task metric (default: logit difference between two answer tokens).

If `R` and `R²` correlate above ~0.9, the distinction collapsed and the project pivots.
The interesting quadrants:

| | low `R` | high `R` |
|---|---|---|
| **high `R²`** | predictable but irreplaceable — redundant representation, unique function | plain redundancy |
| **low `R²`** | genuinely load-bearing and unique | value doesn't matter, only rough presence |

## Positioning (revised after a literature check, July 2026)

The framing changed. It was "build a causal reconstructability hypergraph". That construct
is too easy to read as a relabelling of existing feature-circuit work, and one paper now
occupies the territory directly:

**Cui, Shen & Yang, "SAE Interventions are Unreliable: Post-Intervention Recovery of
Suppressed Behavior"** ([2606.18322](https://arxiv.org/abs/2606.18322), 16 Jun 2026). They
clamp a targeted SAE feature and recover the suppressed behaviour by optimising a residual
perturbation that is encoder-orthogonal, so it cannot simply undo the clamp. 95.8% recovery
on refusal steering. Critically: *"A recovery-path attribution analysis further localizes
this recovery to the SAE reconstruction residual, the component left unexplained by the
SAE."*

That is a published result predicting our core effect should be **weak** — if the
redundancy lives outside the dictionary, other features should not be able to stand in for
an ablated one. It has to be engaged, not ignored.

The mechanisms are genuinely different, which is what leaves room:

| | 2606.18322 | this project |
|---|---|---|
| what is written back | free-form, **encoder-orthogonal** perturbation | `ĝ(S)·W_dec[i]`, along the target's own direction |
| how obtained | per-input adversarial optimisation | learned, input-general function of other features |
| implied locus | outside the dictionary | inside the dictionary |

**So the question becomes: is post-ablation recoverability carried by the dictionary, or by
its residual?** Same experiment, sharper framing, and it makes a recent paper the setup
rather than the competitor. The hypergraph is a downstream artifact if the answer is
"dictionary".

Scope discipline: this is *not* a replication or refutation of 2606.18322. They study
adversarial recovery paths; we ask where a feature's *value* is recoverable from. Related
question, different construct.

**Secondary hypothesis took damage.** [C²R](https://arxiv.org/pdf/2606.30609) (29 Jun 2026)
already frames splitting and absorption as arising from a concept being "inconsistently
distributed across multiple redundant or interfering latents", and penalises co-activation
of directionally similar latents. The redundancy↔splitting link is published, as a training
regulariser rather than a diagnostic. What survives is only that nobody uses *causal*
reconstructability (`R`, not `R²`) as the predictor — a thinner claim than it first looked.

## The `bios` run: the project's central risk is retired

The synthetic task could not answer whether single-feature ablation produces real damage.
The `bios` preset — same GPT-2 small and SAE, but the real Bias-in-Bios professor/nurse
task, class-balanced 50/50 — does. GPT-2 small scores 85.9% against a 50% floor, and the
metric scale drops from ~7.3 to 1.57 (which is why `damage_t` had to replace an absolute
damage threshold).

**All three kill criteria pass**, where the synthetic task failed one outright:

| criterion | synthetic | bios |
|---|---|---|
| `corr(R, R²) > 0.9` → collapsed | 0.217 (pass) | **0.406** (pass) |
| `mean(R − R_const) < 0.05` → const matches learned | **0.007 (FAIL)** | **0.170** (pass) |
| usable targets < 10 | 23 (pass) | **34** (pass) |

Held-out R² also falls from ~0.98 to 0.478, i.e. reconstruction is finally a real problem
rather than a memorisation artifact of a low-entropy task.

**First evidence for the core dissociation.** The scatter shows targets at
`R² = 0.08 → R = 0.79`, `R² = 0.23 → R = 0.89`, `R² = 0.30 → R = 0.90`: features whose
activation value is barely predictable, yet whose causal contribution is almost entirely
restored. Prediction quality and causal restoration come apart, which is the whole premise.

**Caveat that must travel with these numbers.** Damage is statistically strong but small in
absolute terms — |damage| runs 0.007–0.035 on a metric of scale 1.57, with |t| up to 19
because n is 300–800 and the paired difference is consistent. "Restored 94% of the effect"
is therefore 94% of a ~0.008 logit shift. Reliable, modest, and one task on one small model.

## First dictionary-vs-residual numbers

`scripts/08_dictionary_vs_residual.py` fits the same reconstructor from three sources — the
other features in `S`, the SAE error term projected to `|S|` principal components, and both
— then compares prediction and restoration. On the `split` preset, 33/40 targets clearing
the damage noise floor:

| source | mean R² | mean R |
|---|---|---|
| features | 0.780 | 0.964 |
| error term | 0.681 | 0.993 |
| both | 0.866 | 0.997 |

On the `bios` preset, where `R` is finally meaningful, **the sign flips**:

| source | mean R² | mean R |
|---|---|---|
| features | 0.662 | 0.965 |
| error term | **0.754** | 0.979 |
| both | 0.813 | 0.982 |

The dictionary's advantage goes from +0.099 (synthetic) to **−0.092** (real task). On a task
with genuine semantic content, the SAE error term carries *more* information about a
feature's activation than thirty other features do — while restoration works about equally
well from either source.

**This leans against the project's original hypothesis and toward 2606.18322.** The
feature's value is recoverable from both sources, and the dictionary has no privileged
status. Both gaps sit inside the ±0.1 inconclusive band, so nothing is settled — but the
direction of travel is the opposite of what "dictionary-internal redundancy" predicts, and
it reproduces the qualitative claim of the paper we are positioned against.

That is a usable result either way. A careful negative — "post-ablation recoverability is
not carried preferentially by the dictionary" — directly extends a recent paper and is
worth more than a hypergraph built on an effect that is not there. What it must not become
is a hypergraph paper that quietly omits the error-term comparison.

Deleting the error term moves the metric 1.6x as far as ablating one target feature (6.0x
on the synthetic task), confirming the residual carries substantial causal weight in its
own right.

## Formalism

SAE decomposition of the residual stream at layer `L`, final token position:

```
x ≈ b_dec + Σ_j a_j(x) · W_dec[j] + e(x)
```

**Ablation** subtracts the feature's contribution from the *real* residual stream, so the SAE
error term `e(x)` is preserved (this is the Sparse Feature Circuits convention — do not
replace `x` with its SAE reconstruction):

```
x_abl = x − a_i(x) · W_dec[i]
```

**Rescue** writes the predicted activation back along the same direction:

```
x_res = x − a_i(x) · W_dec[i] + ĝ_θ(a_S(x)) · W_dec[i]
      = x + (â_i − a_i) · W_dec[i]
```

**Minimal sufficient set.** For each `f_i`, find the smallest `S ⊂ F \ {f_i}` with `R(S) ≥ τ`
(default `τ = 0.8`). Collecting these gives a hypergraph over features:

```
{f_3, f_7, f_11} ⇒ f_19
```

where `⇒` means "can reconstruct well enough to restore causal function", not "predicts".

## Concrete setup

| | debug config | main config |
|---|---|---|
| model | `gpt2-small` (124M) | `google/gemma-2-2b` |
| SAE | `gpt2-small-res-jb`, `blocks.8.hook_resid_pre` | GemmaScope `layer_12/width_16k/canonical` |
| d_model / d_sae | 768 / 24576 | 2304 / 16384 |
| fits in 8 GB | trivially | yes, bf16, batch ≤ 4 |

**Task.** Bias-in-Bios (`LabHC/bias_in_bios`), the SHIFT/SFC setup — professor vs. nurse.
Framed as a prompted logit-difference so the metric is a clean scalar per prompt. A synthetic
offline task is included so the pipeline can be smoke-tested without downloads.

**Token position.** Everything is measured at the **final prompt token** only — the position the
metric is read from. This is a real limitation (it ignores reconstruction from earlier
positions) and is deliberate for the MVE. Document it; relax it later.

**Feature selection.** Targets `f_i` must have a non-trivial causal effect or `R`'s denominator
is noise. Rank features by indirect effect via attribution patching:

```
IE_i ≈ (a_i^clean − a_i^patch) · ∂m/∂a_i
```

computed with a gradient hook in one backward pass. Take the top ~40 by `|IE|`.

**Reconstructor.** Ridge regression first (closed form, `torch.linalg.lstsq`), MLP second.
Inputs are cached SAE activations of the candidate set at the same position. This is a
regression over a cached matrix, not a training run.

**Minimality search.** Never brute force. Seed `S` from the top ~30 candidates by
|correlation| with `f_i`, then greedy backward elimination: repeatedly drop the feature whose
removal costs the least `R`, stop when `R < τ`. Cost is ~`|S|²/2` rescue evals per target,
i.e. ~10³ total — not `2^n`.

## Controls

A reviewer kills the paper without these. They are implemented as first-class, not extras.

1. **Constant write-back** — inject `f_i`'s dataset-mean activation, ignoring `S` entirely.
   *The most dangerous baseline.* If this rescues as well as `ĝ_θ`, nothing input-dependent
   was learned and the hypergraph is vacuous.
2. **Random-direction rescue** — write `â_i` along a random feature direction at matched norm.
   Rules out rescue being mere norm restoration in the residual stream.
3. **Ablate `S` alone** — if removing the sufficient set independently tanks the metric, `S`
   isn't a reconstructor, it's just the circuit `f_i` lives in.
4. **Held-out prompts** — `ĝ_θ` is fit and evaluated on disjoint splits, always.

## Pre-registered secondary hypothesis

Reconstructable features should be disproportionately those that **split or get absorbed** at
wider SAE widths. GemmaScope ships 16k / 65k / 262k at the same layer, so this is directly
testable: measure `R` at 16k, then check whether high-`R` features are the ones that fragment
at 65k.

If it holds, reconstructability becomes a **diagnostic for SAE feature splitting** — a second
contribution that survives even if the hypergraph claims don't, and an easier sell to the
SAE-evaluation audience.

This is testable on the debug config too: sae-lens ships a `gpt2-small-res-jb-feature-splitting`
release, i.e. the same layer at several widths, which is exactly the comparison this
hypothesis needs and costs almost nothing to run.

**Feasibility gate: passed.** `scripts/07_feature_splitting.py --ladder` measures, per
narrow feature, the best cosine similarity its decoder direction achieves against any
feature of the next width up. Low means the direction was redistributed across finer
features. Across all seven width doublings at `blocks.8.hook_resid_pre`:

| narrow → wide | frac < 0.9 | p05 | p50 | p95 |
|---|---|---|---|---|
| 768 → 1536 | 40.4% | 0.737 | 0.920 | 0.990 |
| 3072 → 6144 | 31.9% | 0.740 | 0.941 | 0.992 |
| 12288 → 24576 | 32.2% | 0.717 | 0.943 | 0.990 |
| 24576 → 49152 | 33.1% | 0.701 | 0.939 | 0.988 |
| 49152 → 98304 | 35.2% | 0.689 | 0.935 | 0.985 |

About a third of features split at every doubling, and the spread is wide. So there is
real variance for `R` to correlate against — the hypothesis is testable, which was not
guaranteed. The rate being near-constant across two orders of magnitude of width is itself
a descriptive finding worth a sentence in any writeup.

**Not yet evidence.** Run on the `split` preset, `corr(R, max-cos) = -0.157` at n=24 —
the predicted sign, but p ≈ 0.46, and every `R` is pinned near 1.0 by the degenerate
synthetic task. This validates the machinery end to end and nothing more. The number only
becomes meaningful once the targets have real causal bite.

**Feature ids are not portable across SAE releases.** Two releases of identical width have
unrelated feature ids, and indexing one's results into the other's decoder lines up
silently. `01_cache_acts.py` stamps each run with its release/id in `run_config.json` and
`07` refuses to correlate on a mismatch.

## Kill criteria

Stated in advance so they can't be rationalised away later.

- `corr(R, R²) > 0.9` → the causal/representational distinction collapsed. Pivot.
- Constant write-back matches `ĝ_θ` rescue → nothing input-dependent learned. Fix the
  reconstructor or the task, don't proceed to the hypergraph.
- Fewer than ~10 targets have measurable `m_clean − m_ablated` → metric too blunt. Sharpen
  the task before adding any machinery.

## Compute

One 8 GB laptop GPU is enough for the MVE.

| stage | cost |
|---|---|
| cache SAE acts, ~10k prompts, 2B model | 1–2 GPU-hours |
| fit reconstructors (40 targets) | minutes, CPU |
| rescue eval + greedy elimination | a few GPU-hours |

Two weeks to the first scatter plot. Everything else is downstream of it.

## Result: the experiment is viable (GPT-2 + Bias-in-Bios)

The `bios` preset — same GPT-2/SAE as debug, real task, balanced classes — **passes all
three kill criteria**:

| | value | threshold | |
|---|---|---|---|
| usable targets | 34 / 40 | ≥ 10 | pass |
| `corr(R, R²)` | 0.406 | < 0.9 | pass |
| `mean(R − R_const)` | 0.170 | > 0.05 | pass |

Predictability explains only ~16% of the variance in causal restorability, which is the
dissociation the whole project rests on. The learned reconstructor beats constant
write-back by a clear margin, so something input-dependent is being used. And the scatter
has a populated low-R²/high-R corner — e.g. a feature at R² = 0.08 with R = 0.79, nearly
unpredictable yet largely restorable.

Caveat to carry forward: `R` itself is compressed near 1.0 for most targets, so most of
the visible structure is in *where the control fails* rather than in spread along the y
axis. Watch whether that persists at Gemma scale.

## Two measurement bugs that inverted the conclusion

Both are worth remembering because each produced a confident, wrong answer.

**The damage gate was an absolute threshold.** `usable()` required `|damage| ≥ 0.1`,
calibrated when the synthetic metric had scale ~7.3. The Bias-in-Bios metric has scale
~1.6, making the same cutoff 6x stricter, and it reported 1 usable target out of 40 —
i.e. "the task is too blunt, abandon it". It is now gated on `damage_t`, the paired
per-row damage over its standard error, which is scale-free and is exactly the quantity
`R` divides by. The same run then yields 34 usable targets with `|t|` up to 25.

The lesson generalises: **damage on this task is small but highly significant.** Effects of
0.03–0.16 on a metric of 1.6 look negligible and are in fact `t` = 10–25 across 800 paired
rows. Never judge these by absolute size.

**"Final token only" was blamed for suppressing damage — wrong.** `08_position_scope.py`
compares ablating a feature at the final token against ablating it everywhere it fires.
Median damage multiplier: **1.0x**. Even a feature with just 9.3% of its activation mass at
the final token gains only 1.1x. The causal path to the metric runs through the final
position because that is where the logits are read, and earlier positions reach it only
through attention. The simplification is sound, and `run_multi`'s KV-cache assumption is
safe.

## What the first debug run actually showed

The `debug` preset (GPT-2 small, synthetic capitals task) runs end to end and **triggers
kill criterion #2**: `mean(R − R_const) = 0.007`. The constant write-back restores just as
much as the learned reconstructor, and greedy elimination happily strips `S` to a single
feature while `R` stays at 0.92.

The cause is visible in the numbers: `damage` maxes at 0.155 against a clean metric of
~7.3. Ablating one feature moves a task this easy by ~2%, so `R` is a ratio of two noise
terms and *anything* written back "restores" it.

This is the smoke test behaving correctly, not a result — but it sharpens a risk for the
main run. **The experiment lives or dies on target selection producing features with real
causal bite.** Before trusting any Bias-in-Bios scatter, check the `damage` column first:
if single-feature ablation moves the metric by a couple of percent there too, no amount of
reconstructor quality will make `R` meaningful, and the task or metric has to get sharper
before the hypergraph work is worth starting.

## Cost, and the optimisation that matters

Greedy elimination is ~`|S|²/2` rescue passes per target — with `n_candidates=30` that is
~435 model passes per target, and it dominates everything else. Three debug targets took
~12 minutes on GPT-2 small.

**Built (`cache.run_multi`).** The edit only ever touches the **final token's** residual at
layer `L`, and under causal attention every earlier position is bit-identical across all
conditions. So the prefix runs once per batch and each condition becomes a single-token
forward on the cached KV. Conditions that share prompts are submitted together: the five
rescue conditions in one call, and a whole greedy round's candidate subsets in another.

Validated by `scripts/00b_check_kv.py`, which pins the fast path to the slow path on real
activations (max abs diff 2e-4) and reports the speedup. Step 05 on three debug targets
went from ~12 minutes to 115 seconds, with a bit-identical greedy trajectory.

The speedup is strongly batch-dependent, because a single-token forward is pure kernel
launch overhead until the batch is wide enough to saturate the GPU:

| batch | 8 | 32 | 128 |
|---|---|---|---|
| speedup | 1.16x | 1.87x | 4.16x |

Hence `batch_size` defaults to 64, not 8. On the main preset the ceiling is VRAM: 5.2 GB of
Gemma weights plus the prefix KV cache. Expect a larger win there than these numbers, since
Bias-in-Bios prompts are ~128 tokens against ~40 here.

## Pipeline

```
00_check_sae.py        verify the SAE is fed the activations it was trained on (gate)
00b_check_kv.py        verify the KV-cache fast path matches the slow path (gate)
01_cache_acts.py       prompts → SAE activations + error term + metric → data/cache/
02_select_features.py  attribution patching → top-|IE| targets → data/targets.json
03_fit_reconstructors  ridge/MLP per target on cached acts → data/reconstructors/
04_rescue_eval.py      rescue + all 4 controls → data/results/rescue.jsonl
05_minimal_sets.py     greedy backward elimination → data/results/hypergraph.json
06_plot.py             the scatter plot → figures/
07_feature_splitting   splitting ladder, and R vs splitting → figures/
08_dictionary_vs_res   where feature information lives → data/sources.jsonl
```
