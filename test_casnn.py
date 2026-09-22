"""
Test suite for CA-SNN-HSI (dual-branch CorrelationAwareSNN)
===========================================================
All tests run on CPU with small synthetic tensors — no dataset files needed.

Test plan
---------
T1  Output shape            – forward pass produces correct (N, num_cls) output.
T2  CACW diagonal           – correlation matrix built inside CACW has diagonal=1.
T3  Spectral branch gating  – CACW gates actually change feature values.
T4  Spectral redundancy     – non-uniform channel weights (differential weighting).
T5  Pearson adjacency       – symmetric, non-negative, correct shape.
T6  Spatial branch shape    – spatial branch outputs (T, N, C).
T7  Fusion weights          – cross-branch β sums to 1.
T8  Gradient flow           – both branches receive gradients.
T9  No NaN / Inf            – finite outputs for normal inputs.
T10 Parameter counts        – sensible counts for tiny and realistic configs.
T11 Determinism             – same input → same output in eval mode.
T12 Multi-dataset configs   – PU, HU, WHLK all run without error.
"""

import sys
import traceback
import torch
import torch.nn.functional as F

import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from CASNN import (
    CACW,
    SpectralBranch,
    SpatialBranch,
    CorrelationAwareSNN,
    _pearson_adj,
)
from spikingjelly.activation_based import functional


def _set_multi_step(module):
    """Set a standalone module to multi-step mode (required for layer.BN/LIF)."""
    functional.set_step_mode(module, step_mode='m')


# ── helpers ───────────────────────────────────────────────────────────────────

def _pass(name):
    print(f"  PASS  {name}")


def _fail(name, msg):
    print(f"  FAIL  {name}: {msg}")
    raise AssertionError(f"{name}: {msg}")


def run(name, fn):
    try:
        fn()
        _pass(name)
        return True
    except Exception as exc:
        _fail(name, str(exc))
        traceback.print_exc()
        return False


def _tiny_model():
    return CorrelationAwareSNN(
        T=2, img_size=5, num_cls=4, input_dim=8,
        hidden=12, n_stages=2, gcn_layers=1, K_hop=1,
        use_cupy=False,
    ).eval()


def _batch(N=4, img_size=5, S=8):
    return torch.randn(N, img_size * img_size, S)


# ══════════════════════════════════════════════════════════════════════════════

def test_t1_output_shape():
    """T1: forward produces (N, num_cls)."""
    model = _tiny_model()
    x = _batch()
    with torch.no_grad():
        out = model(x)
    assert out.shape == (4, 4), f"expected (4,4) got {out.shape}"


def test_t2_cacw_diagonal_is_one():
    """T2: CACW builds a Pearson correlation matrix — diagonal must equal 1."""
    cacw = CACW(n=8, d=6)
    captured = {}
    def hook(mod, inp, out):
        captured["C_tilde"] = inp[0].detach()    # fc1 input = C_tilde
    cacw.fc1.register_forward_hook(hook)

    X = torch.randn(3, 20, 8)     # [batch, m=20, n=8]
    cacw(X)
    C = captured["C_tilde"]        # [3, 8, 8]
    diag = C.diagonal(dim1=-2, dim2=-1)
    assert torch.allclose(diag, torch.ones_like(diag), atol=1e-5), \
        f"diagonal not 1: {diag}"


def test_t3_spectral_branch_gating():
    """T3: channel-correlation gates in SpectralBranch actually modify features."""
    torch.manual_seed(1)
    branch = SpectralBranch(hidden=12, n_stages=2)
    _set_multi_step(branch)
    branch.eval()
    T, N, L2, C = 2, 4, 25, 12
    x = torch.randn(T, N, L2, C)

    with torch.no_grad():
        out = branch(x)

    assert out.shape == (T, N, C)
    # Must differ from a plain spatial mean (gating adds value)
    plain = x.mean(dim=-2)    # [T, N, C]
    assert not torch.allclose(out, plain, atol=1e-3), \
        "SpectralBranch output == plain spatial mean → gating had no effect"
    assert torch.isfinite(out).all()


def test_t4_spectral_nonuniform_channel_weights():
    """T4: CACW inside SpectralBranch produces non-uniform weights across channels."""
    torch.manual_seed(42)
    T, N, L2, C = 2, 6, 25, 12
    cacw = CACW(n=C, d=9)
    x = torch.randn(T * N, L2, C)
    with torch.no_grad():
        gamma = cacw(x)    # [T·N, C]

    var = gamma.var(dim=-1).mean()
    assert var > 1e-6, \
        f"CACW channel weights are nearly uniform (var={var:.2e}) — no differentiation"


def test_t5_pearson_adj_properties():
    """T5: _pearson_adj is symmetric, non-negative, and has the right shape."""
    torch.manual_seed(2)
    T, N, L2, C = 2, 3, 9, 12
    x = torch.randn(T, N, L2, C)
    adj = _pearson_adj(x)

    assert adj.shape == (T, N, L2, L2), f"wrong shape {adj.shape}"
    assert (adj >= 0).all(), "adj has negative values"
    sym_err = (adj - adj.transpose(-2, -1)).abs().max().item()
    assert sym_err < 1e-5, f"adj not symmetric (max diff {sym_err:.2e})"
    assert torch.isfinite(adj).all()


def test_t6_spatial_branch_shape():
    """T6: SpatialBranch outputs (T, N, C)."""
    torch.manual_seed(3)
    branch = SpatialBranch(hidden=12, n_stages=2, gcn_layers=1, K_hop=1)
    _set_multi_step(branch)
    branch.eval()
    T, N, L2, C = 2, 4, 25, 12
    x = torch.randn(T, N, L2, C)
    with torch.no_grad():
        out = branch(x)
    assert out.shape == (T, N, C), f"got {out.shape}"
    assert torch.isfinite(out).all()


def test_t7_fusion_weights_sum_to_one():
    """T7: cross-branch β from CACW+softmax sums to 1."""
    torch.manual_seed(4)
    C = 16
    fusion_cacw = CACW(n=2, d=4)
    T, N = 2, 5
    f_spec = torch.randn(T, N, C)
    f_spat = torch.randn(T, N, C)
    branches = torch.stack([f_spec, f_spat], dim=-1)   # [T, N, C, 2]
    with torch.no_grad():
        beta = F.softmax(
            fusion_cacw(branches.reshape(T * N, C, 2)).reshape(T, N, 2),
            dim=-1,
        )
    sums = beta.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), \
        f"fusion weights do not sum to 1: {sums}"


def test_t8_gradient_flow():
    """T8: both spectral and spatial branches receive gradients."""
    model = _tiny_model()
    model.train()
    x = _batch()
    model(x).sum().backward()

    spec = [p.grad for n, p in model.named_parameters()
            if "spectral_branch" in n and p.grad is not None]
    spat = [p.grad for n, p in model.named_parameters()
            if "spatial_branch" in n and p.grad is not None]
    assert spec, "no gradients reached spectral_branch"
    assert spat, "no gradients reached spatial_branch"

    no_grad = [n for n, p in model.named_parameters()
               if p.requires_grad and p.grad is None
               and "running" not in n and "num_batches" not in n]
    assert not no_grad, f"leaf params with no grad: {no_grad}"


def test_t9_no_nan_inf():
    """T9: finite output and gradients for normal random inputs."""
    model = _tiny_model()
    model.train()
    x = _batch()
    out = model(x)
    assert not out.isnan().any(), "NaN in output"
    assert not out.isinf().any(), "Inf in output"
    out.sum().backward()
    for n, p in model.named_parameters():
        if p.grad is not None:
            assert not p.grad.isnan().any(), f"NaN grad in {n}"
            assert not p.grad.isinf().any(), f"Inf grad in {n}"


def test_t10_parameter_counts():
    """T10: sensible parameter counts for tiny and PU-realistic configs."""
    tiny = _tiny_model()
    n_tiny = sum(p.numel() for p in tiny.parameters())
    assert 100 < n_tiny < 500_000, f"tiny params={n_tiny:,}"
    print(f"\n         tiny : {n_tiny:,}")

    pu = CorrelationAwareSNN(
        T=3, img_size=19, num_cls=9, input_dim=103,
        hidden=32, n_stages=4, gcn_layers=2, K_hop=2, use_cupy=False,
    )
    n_pu = sum(p.numel() for p in pu.parameters())
    assert 10_000 < n_pu < 10_000_000, f"PU params={n_pu:,}"
    print(f"         PU   : {n_pu:,}")


def test_t11_determinism():
    """T11: same input gives same output in eval mode."""
    model = _tiny_model()
    model.eval()
    x = torch.randn(4, 25, 8)
    with torch.no_grad():
        o1 = model(x).clone()
        o2 = model(x).clone()
    assert torch.allclose(o1, o2, atol=1e-6), \
        "two forward passes differ — non-deterministic eval"


def test_t12_multi_dataset_configs():
    """T12: model runs for PU, HU, WHLK configurations."""
    configs = [
        dict(name="PU",   img_size=19, num_cls=9,  input_dim=103,
             hidden=32, n_stages=4, gcn_layers=2, K_hop=2),
        dict(name="HU",   img_size=13, num_cls=15, input_dim=144,
             hidden=96, n_stages=4, gcn_layers=2, K_hop=1),
        dict(name="WHLK", img_size=15, num_cls=9,  input_dim=270,
             hidden=48, n_stages=4, gcn_layers=2, K_hop=1),
    ]
    for cfg in configs:
        name = cfg.pop("name")
        model = CorrelationAwareSNN(T=3, use_cupy=False, **cfg).eval()
        x = torch.randn(2, cfg["img_size"] ** 2, cfg["input_dim"])
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, cfg["num_cls"]), \
            f"{name}: wrong shape {out.shape}"
        assert torch.isfinite(out).all(), f"{name}: NaN/Inf in output"
        n = sum(p.numel() for p in model.parameters())
        print(f"\n         {name}: {n:,} params")


# ══════════════════════════════════════════════════════════════════════════════

TESTS = [
    ("T1  Output shape",                    test_t1_output_shape),
    ("T2  CACW diagonal = 1",               test_t2_cacw_diagonal_is_one),
    ("T3  Spectral branch gating",          test_t3_spectral_branch_gating),
    ("T4  Non-uniform channel weights",     test_t4_spectral_nonuniform_channel_weights),
    ("T5  Pearson adjacency properties",    test_t5_pearson_adj_properties),
    ("T6  Spatial branch output shape",     test_t6_spatial_branch_shape),
    ("T7  Fusion weights sum to 1",         test_t7_fusion_weights_sum_to_one),
    ("T8  Gradient flow both branches",     test_t8_gradient_flow),
    ("T9  No NaN / Inf",                    test_t9_no_nan_inf),
    ("T10 Parameter counts",                test_t10_parameter_counts),
    ("T11 Determinism",                     test_t11_determinism),
    ("T12 Multi-dataset configs",           test_t12_multi_dataset_configs),
]


def main():
    print("\n" + "=" * 60)
    print("  CA-SNN-HSI (dual-branch)  –  Test Suite")
    print("=" * 60)

    passed = failed = 0
    for label, fn in TESTS:
        print(f"\n[{label}]")
        if run(label, fn):
            passed += 1
        else:
            failed += 1

    print("\n" + "=" * 60)
    print(f"  {passed} passed / {failed} failed / {len(TESTS)} total")
    print("=" * 60 + "\n")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
