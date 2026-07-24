"""Central configuration. Two presets: a cheap debug config and the main config."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
FIGURES = ROOT / "figures"


@dataclass
class Config:
    name: str = "debug"

    # model + SAE
    model_name: str = "gpt2"
    sae_release: str = "gpt2-small-res-jb"
    sae_id: str = "blocks.8.hook_resid_pre"
    # Index into the decoder-layer list whose *output* we hook. SAEs trained on
    # `blocks.N.hook_resid_pre` see the output of layer N-1, so this is sae_layer - 1.
    # SAEs trained on resid_post of layer N use hook_layer = N.
    hook_layer: int = 7
    # Mean-centre the residual along d_model before encoding. Required for SAEs trained on
    # TransformerLens activations (its `center_writing_weights`); wrong for SAEs trained on
    # raw HF activations. Verify with scripts/00_check_sae.py before trusting a new SAE.
    center_resid: bool = True

    dtype: str = "float32"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # task
    task: str = "synthetic"  # "synthetic" | "bias_in_bios"
    n_prompts: int = 512
    max_len: int = 128
    # Bigger batches matter more than usual: run_multi's per-condition cost is a
    # single-token forward, which is pure launch overhead until the batch is wide enough
    # to saturate the GPU. Measured on GPT-2 small: 1.16x speedup at 8, 4.16x at 128.
    batch_size: int = 64
    train_frac: float = 0.6  # rest is held out for R^2 and rescue

    # feature selection
    n_targets: int = 40
    min_active_frac: float = 0.02  # target must fire on >=2% of prompts

    # reconstruction
    n_candidates: int = 30  # seed set size for S, by |corr| with the target
    ridge_lambda: float = 1.0
    reconstructor: str = "ridge"  # "ridge" | "mlp"

    # minimality
    tau: float = 0.8  # rescue threshold for "sufficient"

    seed: int = 0

    def __post_init__(self) -> None:
        self.run_dir = DATA / self.name
        self.run_dir.mkdir(parents=True, exist_ok=True)

    @property
    def torch_dtype(self) -> torch.dtype:
        return {"float32": torch.float32, "bfloat16": torch.bfloat16}[self.dtype]

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("run_dir", None)
        return d


DEBUG = Config()

MAIN = Config(
    name="main",
    model_name="google/gemma-2-2b",
    sae_release="gemma-scope-2b-pt-res-canonical",
    sae_id="layer_12/width_16k/canonical",
    hook_layer=12,  # GemmaScope res SAEs are trained on resid_post of this layer
    center_resid=False,  # GemmaScope trained on raw activations -- CONFIRM with 00_check_sae
    dtype="bfloat16",
    task="bias_in_bios",
    n_prompts=10_000,
    batch_size=16,  # 8 GB is the constraint: 5.2 GB of weights plus the prefix KV cache
)

# Same model and hook point as DEBUG, but from the release that ships one SAE per width.
# Required for the feature-splitting analysis: feature ids must come from the same release
# whose decoder the splitting score is computed against.
SPLIT = Config(
    name="split",
    sae_release="gpt2-small-res-jb-feature-splitting",
    sae_id="blocks.8.hook_resid_pre_24576",
    hook_layer=7,
)

# The cheap way to ask the question that actually gates the project: does single-feature
# ablation on a real task produce measurable damage? Same model/SAE as DEBUG, real task.
# If damage is still ~2% of the metric here, the problem is the task, not the model.
BIOS = Config(
    name="bios",
    task="bias_in_bios",
    n_prompts=2000,
    batch_size=32,
)

PRESETS = {"debug": DEBUG, "main": MAIN, "split": SPLIT, "bios": BIOS}


def get_config(name: str = "debug") -> Config:
    if name not in PRESETS:
        raise KeyError(f"unknown preset {name!r}; have {list(PRESETS)}")
    # re-instantiate so run_dir is created fresh
    return Config(**PRESETS[name].to_dict())
