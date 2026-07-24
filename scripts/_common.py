"""Shared bootstrap for the pipeline scripts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rnar import get_config  # noqa: E402
from rnar import data, model  # noqa: E402
from rnar.config import PRESETS  # noqa: E402


def parse(description: str, **extra) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--preset", default="debug", choices=sorted(PRESETS))
    for flag, kwargs in extra.items():
        p.add_argument(f"--{flag}", **kwargs)
    return p.parse_args()


def setup(args):
    """Config + loaded model/SAE bundle + task. Prints the run header."""
    cfg = get_config(args.preset)
    for key in vars(args):
        if key != "preset" and getattr(args, key) is not None and hasattr(cfg, key):
            setattr(cfg, key, getattr(args, key))

    print(f"[run] preset={cfg.name} model={cfg.model_name} sae={cfg.sae_id} "
          f"layer={cfg.hook_layer} task={cfg.task} device={cfg.device}")
    bundle = model.load(cfg)
    task = data.build(cfg, bundle.tokenizer)
    print(f"[run] d_sae={bundle.d_sae} d_model={bundle.d_model} n_prompts={len(task)}")
    return cfg, bundle, task


def path(cfg, *parts: str) -> Path:
    return cfg.run_dir.joinpath(*parts)
