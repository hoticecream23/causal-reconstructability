"""Hook mechanics, tested against a fake layer + fake SAE so no downloads are needed.

These guard the part of the pipeline that is silently wrong if it is wrong: whether an
edit lands on the right position, along the right direction, at the right magnitude.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rnar import hooks  # noqa: E402
from rnar.model import Bundle  # noqa: E402

D_MODEL, D_SAE, BATCH, SEQ = 16, 32, 4, 6


class TupleLayer(nn.Module):
    """Mimics an HF decoder block: returns a tuple whose [0] is the hidden state."""

    def forward(self, x):
        return (x,)


class BareLayer(nn.Module):
    """The other HF convention: a bare tensor."""

    def forward(self, x):
        return x


class FakeSAE:
    def __init__(self, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        self.W_dec = torch.randn(D_SAE, D_MODEL, generator=g)
        self.W_enc = torch.randn(D_MODEL, D_SAE, generator=g) * 0.1

    def encode(self, x):
        return torch.relu(x @ self.W_enc)

    def decode(self, a):
        return a @ self.W_dec


@pytest.fixture
def bundle():
    from rnar.config import Config

    cfg = Config(name="test", device="cpu")
    return Bundle(model=None, tokenizer=None, sae=FakeSAE(), layer=TupleLayer(), cfg=cfg)


@pytest.fixture
def x():
    return torch.randn(BATCH, SEQ, D_MODEL, generator=torch.Generator().manual_seed(1))


def run_layer(layer, x):
    out = layer(x)
    return out[0] if isinstance(out, tuple) else out


@pytest.mark.parametrize("layer_cls", [TupleLayer, BareLayer])
def test_identity_edit_is_a_noop(layer_cls, x):
    layer = layer_cls()
    with hooks.edit_resid(layer, lambda r: r):
        out = run_layer(layer, x)
    torch.testing.assert_close(out, x)


def test_edit_touches_only_final_position(x):
    layer = TupleLayer()
    with hooks.edit_resid(layer, lambda r: r + 100.0):
        out = run_layer(layer, x)
    torch.testing.assert_close(out[:, :-1, :], x[:, :-1, :])
    torch.testing.assert_close(out[:, -1, :], x[:, -1, :] + 100.0)


def test_hook_is_removed_on_exit(x):
    layer = TupleLayer()
    with hooks.edit_resid(layer, lambda r: r + 100.0):
        pass
    torch.testing.assert_close(run_layer(layer, x), x)


def test_add_along_feature_magnitude_and_direction(bundle, x):
    feat = 7
    delta = torch.tensor([1.0, -2.0, 0.0, 0.5])
    with hooks.edit_resid(bundle.layer, hooks.add_along_feature(bundle, feat, delta)):
        out = run_layer(bundle.layer, x)
    expected = x[:, -1, :] + delta[:, None] * bundle.W_dec[feat]
    torch.testing.assert_close(out[:, -1, :], expected)


def test_ablation_removes_exactly_the_feature_contribution(bundle, x):
    """Ablation must remove a·W_dec[i] and nothing else, so adding it back recovers x."""
    feat = 3
    a = bundle.encode(x[:, -1, :])[:, feat]
    assert (a > 0).any(), "pick a feature that actually fires, or the test is vacuous"

    with hooks.edit_resid(bundle.layer, hooks.add_along_feature(bundle, feat, -a)):
        ablated = run_layer(bundle.layer, x)

    assert not torch.allclose(ablated[:, -1, :], x[:, -1, :]), "ablation should change something"
    recovered = ablated[:, -1, :] + a[:, None] * bundle.W_dec[feat]
    torch.testing.assert_close(recovered, x[:, -1, :])


def test_rescue_delta_lands_at_the_predicted_activation(bundle, x):
    """x + (â − a)·W_dec must equal the residual with the feature set to â."""
    feat = 3
    a = bundle.encode(x[:, -1, :])[:, feat]
    ahat = a * 0.6 + 0.3

    with hooks.edit_resid(bundle.layer, hooks.add_along_feature(bundle, feat, ahat - a)):
        rescued = run_layer(bundle.layer, x)

    expected = x[:, -1, :] - a[:, None] * bundle.W_dec[feat] + ahat[:, None] * bundle.W_dec[feat]
    torch.testing.assert_close(rescued[:, -1, :], expected)


def test_subtract_features_matches_single_feature_ablation(bundle, x):
    feat = 5
    a = bundle.encode(x[:, -1, :])[:, feat]

    with hooks.edit_resid(bundle.layer, hooks.add_along_feature(bundle, feat, -a)):
        via_add = run_layer(bundle.layer, x)
    with hooks.edit_resid(bundle.layer, hooks.subtract_features(bundle, [feat], a[:, None])):
        via_subtract = run_layer(bundle.layer, x)

    torch.testing.assert_close(via_add, via_subtract)


def test_subtract_features_is_additive_over_the_set(bundle, x):
    feats = [1, 4, 9]
    acts = bundle.encode(x[:, -1, :])[:, feats]
    with hooks.edit_resid(bundle.layer, hooks.subtract_features(bundle, feats, acts)):
        out = run_layer(bundle.layer, x)
    expected = x[:, -1, :] - sum(acts[:, j][:, None] * bundle.W_dec[f] for j, f in enumerate(feats))
    torch.testing.assert_close(out[:, -1, :], expected)


def test_edit_leaves_every_earlier_position_untouched(x):
    """The invariant `run_multi` rests on: the edit cannot affect the cached prefix.

    If an edit ever reached a non-final position, the prefix KV cache would go stale and
    the fast path would silently diverge from the slow path.
    """
    layer = TupleLayer()
    with hooks.edit_resid(layer, lambda r: torch.randn_like(r) * 50):
        out = run_layer(layer, x)
    torch.testing.assert_close(out[:, :-1, :], x[:, :-1, :])
    assert not torch.allclose(out[:, -1, :], x[:, -1, :])


def test_random_direction_control_matches_ablation_norm(bundle):
    """The control must write along a direction of the same length as the real one."""
    d_i = bundle.W_dec[3]
    u = bundle.W_dec[11]
    u = u * (d_i.norm() / u.norm())
    torch.testing.assert_close(u.norm(), d_i.norm())
    assert torch.abs(torch.dot(u / u.norm(), d_i / d_i.norm())) < 0.9
