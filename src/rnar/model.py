"""Model + SAE loading, and resolution of the decoder-layer module we hook."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import Config


@dataclass
class Bundle:
    model: torch.nn.Module
    tokenizer: object
    sae: object
    layer: torch.nn.Module
    cfg: Config

    @property
    def W_dec(self) -> torch.Tensor:
        """[d_sae, d_model], float32."""
        return self.sae.W_dec

    @property
    def d_sae(self) -> int:
        return self.sae.W_dec.shape[0]

    @property
    def d_model(self) -> int:
        return self.sae.W_dec.shape[1]

    def encode(self, resid: torch.Tensor) -> torch.Tensor:
        """Residual stream -> SAE feature activations. Always float32.

        SAEs trained on TransformerLens activations saw a residual stream that had been
        mean-centred along d_model (TL's `center_writing_weights`). Raw HF activations are
        not centred, and the offset grows with depth: feeding them uncentred to a GPT-2
        res-jb SAE at layer 7 gives L0 625 / explained variance -2.07 instead of
        L0 74 / +0.91. Run `scripts/00_check_sae.py` to confirm the flag for a new SAE.

        Only encoding is affected. Decoder rows are mean-zero to within 4% of their norm,
        and the all-ones component is removed by the next LayerNorm regardless, so
        ablation and injection still write into the raw stream unchanged.
        """
        return self.split_sae(resid)[0]

    def split_sae(self, resid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Residual -> (feature activations, SAE error term), both float32.

        The error term is what the dictionary fails to explain. Keeping it separately is
        what lets us ask where a feature's information actually lives: in the other
        features, or in the part of the residual the SAE never captured.
        """
        x = resid.to(torch.float32)
        if self.cfg.center_resid:
            x = x - x.mean(dim=-1, keepdim=True)
        a = self.sae.encode(x)
        return a, x - self.sae.decode(a)


def get_layers(model: torch.nn.Module) -> torch.nn.ModuleList:
    """The list of decoder blocks, across the two HF layouts we care about."""
    for path in ("model.layers", "transformer.h"):
        obj = model
        for part in path.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if obj is not None:
            return obj
    raise AttributeError(f"could not locate decoder layers on {type(model).__name__}")


def load(cfg: Config) -> Bundle:
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    # Left padding puts the metric-bearing final token at index -1 for every row,
    # which is what the entire caching/hook pipeline assumes.
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # transformers >= 4.56 renamed `torch_dtype` to `dtype`.
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name, dtype=cfg.torch_dtype)
    model.to(cfg.device)
    model.eval()
    model.requires_grad_(False)

    sae = _load_sae(cfg)

    layers = get_layers(model)
    if not 0 <= cfg.hook_layer < len(layers):
        raise IndexError(f"hook_layer={cfg.hook_layer} out of range for {len(layers)} layers")

    return Bundle(model=model, tokenizer=tokenizer, sae=sae, layer=layers[cfg.hook_layer], cfg=cfg)


def _load_sae(cfg: Config):
    from sae_lens import SAE

    out = SAE.from_pretrained(cfg.sae_release, cfg.sae_id, device=cfg.device)
    sae = out[0] if isinstance(out, tuple) else out  # returned a triple before sae-lens v5
    sae = sae.to(torch.float32)
    sae.eval()
    for p in sae.parameters():
        p.requires_grad_(False)
    return sae
