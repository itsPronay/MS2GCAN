"""
Correlation-Aware Spiking Neural Network for Hyperspectral Image Classification
================================================================================
CA-SNN-HSI  ·  dual-branch

Two parallel branches share a common spectral projection, then diverge:

  Spectral Branch  — channel-axis correlation is the processing engine
      At every stage a CACW module builds the C×C Pearson correlation
      matrix of the C hidden channels (treating the L² spatial pixels as
      observations).  A learned MLP maps each row of that matrix to a
      sigmoid gate α ∈ (0,1)^C.  Redundant channels (high r_ij) are
      suppressed; uniquely-firing channels (low r_ij) are amplified.
      Only after this intra-stage gating do features enter a
      SpikingResBlock.  Correlation IS the selection mechanism — it is
      not an add-on.

  Spatial Branch   — pixel-axis correlation is the processing engine
      At every stage an adaptive Pearson adjacency matrix A[L²×L²] is
      built from the current hidden features: pixels whose spike-firing
      profiles are positively correlated receive a strong edge.  GCN
      layers then perform correlation-driven spatial message passing,
      followed by a SpikingResBlock.  The adjacency is re-estimated at
      every stage so the graph topology co-evolves with the features.
      Correlation IS the graph — it is not an add-on.

  Cross-Branch Fusion
      The two branch outputs (each [T, N, C]) are stacked into a [T·N,
      C, 2] tensor and fed to a CACW module with n=2: the 2×2 Pearson
      correlation between the two branch feature vectors acts as a
      "uniqueness detector" — if the branches are highly correlated they
      carry redundant information and are blended equally; if they are
      orthogonal both are amplified.  Softmax over the resulting β ∈ ℝ²
      produces a learned blend.

Pipeline
--------
  Input [N, L², S]
    ↓  temporal expand → [T, N, L², S]
    ↓  shared spectral projection (FC→BN→LIF ×2)
  [T, N, L², C]
    ├─── Spectral Branch ─────────────────────────────────────────────────
    │     for k in 1…K:
    │       α = sigmoid(CACW_k(F))     # C×C channel correlation → gates
    │       F = F ⊙ α                  # intra-stage spectral reweighting
    │       F = SpikingResBlock_k(F)   # spiking feature refinement
    │     spatial mean-pool → [T, N, C]
    │
    ├─── Spatial Branch ──────────────────────────────────────────────────
    │     for k in 1…K:
    │       A = PearsonAdj(F)          # L²×L² pixel-correlation graph
    │       F = GCN_k(F, A) + F       # correlation-driven message passing
    │       F = SpikingResBlock_k(F)
    │     K-hop centre aggregation → [T, N, C]
    │
    └─── Cross-Branch Fusion (CACW over 2 branches)
          β = softmax(CACW_fusion(stack([F_spec, F_spat])))   # ∈ ℝ²
          F = β₀·F_spec + β₁·F_spat   → [T, N, C]
          temporal softmax weighting   → [N, C]
          linear classifier            → [N, num_cls]

References
----------
Correlation-Aware Covariance Weighting (CACW) is adapted from:
  "A General Adaptive Dual-level Weighting Mechanism for Remote Sensing
   Pansharpening", CVPR 2025.
  Re-derived here for spiking binary/residual features at channel-,
  stage-, and spatial-axes.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import neuron, functional, surrogate, layer


# ─────────────────────────────────────────────────────────────────────────────
#  Module-level helper: adaptive Pearson adjacency
# ─────────────────────────────────────────────────────────────────────────────

def _pearson_adj(spike: torch.Tensor) -> torch.Tensor:
    """
    Build a symmetric, degree-normalised Pearson similarity graph over L² pixels.

    Parameters
    ----------
    spike : Tensor[T, N, L², C]

    Returns
    -------
    adj : Tensor[T, N, L², L²]
    """
    mu       = spike.mean(dim=-1, keepdim=True)          # mean over C
    centered = spike - mu
    # eps guards zero-variance pixels (zero-padded patch borders)
    normed   = F.normalize(centered, p=2, dim=-1, eps=1e-8)
    sim      = torch.matmul(normed, normed.transpose(-2, -1))
    adj      = F.relu(sim)                               # keep positive correlations
    deg      = adj.sum(-1, keepdim=True)
    d_inv    = torch.pow(deg + 1e-6, -0.5)
    return adj * d_inv * d_inv.transpose(-2, -1)         # symmetric normalisation


# ─────────────────────────────────────────────────────────────────────────────
#  Core primitive: Correlation-Aware Covariance Weighting (CACW)
# ─────────────────────────────────────────────────────────────────────────────

class CACW(nn.Module):
    """
    Correlation-Aware Covariance Weighting.

    Maps a batch of observations X ∈ ℝ^{…×m×n} to per-feature
    importance weights γ ∈ ℝ^{…×n}.

    Steps
    -----
    1. Mean-centre X along the sample axis (dim -2).
    2. L2-normalise each column of X_c along the sample axis.
    3. Compute Pearson correlation matrix C̃ = X_c_normed^T · X_c_normed.
       Diagonal entries = 1; off-diagonal ∈ (-1, 1).
    4. Row-wise two-layer MLP:
           row_i  →  Linear(n, d) → LeakyReLU → Linear(d, 1)
       yielding γ ∈ ℝ^n.

    Args
    ----
    n : number of features (columns).
    d : MLP hidden width.
    """

    def __init__(self, n: int, d: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(n, d, bias=True)
        self.act  = nn.LeakyReLU(negative_slope=0.1, inplace=False)
        self.fc2  = nn.Linear(d, 1, bias=False)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """X : [..., m, n]  →  gamma : [..., n]"""
        Xc        = X - X.mean(dim=-2, keepdim=True)
        Xc_normed = F.normalize(Xc, p=2, dim=-2, eps=1e-8)
        C_tilde   = Xc_normed.transpose(-2, -1) @ Xc_normed   # [..., n, n]
        h         = self.act(self.fc1(C_tilde))                # [..., n, d]
        return self.fc2(h).squeeze(-1)                         # [..., n]


# ─────────────────────────────────────────────────────────────────────────────
#  Auxiliary spiking layers
# ─────────────────────────────────────────────────────────────────────────────

class SpikingResBlock(nn.Module):
    """FC → BN → LIF with additive skip.  Operates on [T, N, L², C]."""

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.fc  = layer.Linear(hidden, hidden)
        self.bn  = layer.BatchNorm1d(hidden)
        self.lif = neuron.LIFNode(
            decay_input=False, detach_reset=True,
            surrogate_function=surrogate.ATan(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        out = self.fc(x)
        out = self.bn(out.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        return self.lif(out) + res


class GraphConvLayer(nn.Module):
    """Symmetric graph convolution  X' = A · X · W."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.linear = layer.Linear(in_dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        return torch.matmul(adj, self.linear(x))


class KHopCenterAggregator(nn.Module):
    """
    K-hop neighbourhood aggregation for the centre pixel.

    Computes A^k row masks (k = 0…K) for the centre pixel, gathers
    per-hop feature vectors by weighted pooling, then fuses them via a
    SiLU-gated softmax.
    """

    def __init__(self, K: int = 2, hidden: int = 64, eps: float = 1e-6) -> None:
        super().__init__()
        self.K   = K
        self.eps = eps
        self.linear = layer.Linear(hidden, hidden)
        self.gate = nn.Sequential(
            nn.LayerNorm(hidden * (K + 1)),
            layer.Linear(hidden * (K + 1), hidden),
            nn.SiLU(),
            layer.Linear(hidden, K + 1),
            nn.Softmax(dim=-1),
        )

    def forward(
        self, gcn_out: torch.Tensor, adj: torch.Tensor, center_idx: int
    ) -> torch.Tensor:
        gcn_out = self.linear(gcn_out)

        if self.K == 0:
            return gcn_out[:, :, center_idx, :]

        T, N, L2, H = gcn_out.shape
        device = adj.device

        adj_flat   = adj.reshape(T * N, L2, L2)
        eye        = torch.eye(L2, device=device).unsqueeze(0).expand(T * N, -1, -1)
        adj_powers = [eye.clone()]
        cur        = adj_flat.clone()
        for k in range(1, self.K + 1):
            adj_powers.append(cur)
            if k < self.K:
                cur = torch.bmm(cur, adj_flat)

        gcn_flat = gcn_out.reshape(T * N, L2, H)
        feats: list[torch.Tensor] = []
        for ap in adj_powers:
            mask = ap[:, center_idx, :]
            mask = mask / (mask.sum(-1, keepdim=True) + self.eps)
            feats.append(torch.bmm(mask.unsqueeze(1), gcn_flat).squeeze(1))

        feat_cat   = torch.cat(feats,   dim=-1)
        feat_stack = torch.stack(feats, dim=-1)
        weights    = self.gate(feat_cat).unsqueeze(-2)
        fused      = (weights * feat_stack).sum(-1)
        return fused.view(T, N, H)


# ─────────────────────────────────────────────────────────────────────────────
#  Spectral Branch
# ─────────────────────────────────────────────────────────────────────────────

class SpectralBranch(nn.Module):
    """
    Spectral-correlation-aware spiking branch.

    Channel correlation is the **core processing mechanism** in this branch.
    At every stage, a CACW module reads the C×C Pearson correlation matrix
    of the C hidden channels (using the L² spatial positions as observations)
    and outputs per-channel sigmoid gates α ∈ (0,1)^C.  Channels that
    co-fire heavily with many others (spectral redundancy) are suppressed;
    channels with unique firing patterns are amplified.  This gating happens
    before the SpikingResBlock — correlation drives what the block even sees.

    Stage pipeline (repeated n_stages times):
        F [T, N, L², C]
        → CACW(F) → α ∈ (0,1)^C          channel importance from C×C corr
        → F ⊙ α                           intra-stage spectral reweighting
        → SpikingResBlock(F)              spiking feature refinement

    Output: spatial mean-pool over L² → [T, N, C]
    """

    def __init__(self, hidden: int, n_stages: int, cacw_ratio: float = 0.8) -> None:
        super().__init__()
        d = max(4, int(cacw_ratio * hidden))
        # One CACW gate per stage — correlation selects what each stage processes
        self.cacw_gates = nn.ModuleList([CACW(n=hidden, d=d) for _ in range(n_stages)])
        self.stages     = nn.ModuleList([SpikingResBlock(hidden) for _ in range(n_stages)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : [T, N, L², C]  →  [T, N, C]"""
        T, N, L2, C = x.shape
        for cacw, stage in zip(self.cacw_gates, self.stages):
            # ── Channel-correlation gating ───────────────────────────────────
            xr    = x.reshape(T * N, L2, C)         # [T·N, L², C]  m=L², n=C
            alpha = torch.sigmoid(cacw(xr))          # [T·N, C]
            # sigmoid bounds to (0,1): no sign flips on spike-residual features
            alpha = alpha.reshape(T, N, 1, C)        # broadcast over L²
            x     = x * alpha                        # channel reweighting
            # ── Spiking feature refinement ───────────────────────────────────
            x     = stage(x)
        return x.mean(dim=-2)                        # spatial pool → [T, N, C]


# ─────────────────────────────────────────────────────────────────────────────
#  Spatial Branch
# ─────────────────────────────────────────────────────────────────────────────

class SpatialBranch(nn.Module):
    """
    Spatial-correlation-aware spiking branch.

    Pixel-to-pixel correlation is the **core processing mechanism** in this
    branch.  At every stage the Pearson correlation between every pair of
    the L² patch pixels (measured across the C hidden channels) is computed
    and used directly as the graph adjacency matrix.  GCN layers perform
    message passing along those correlation edges, then a SpikingResBlock
    refines the features.  The adjacency is re-estimated from the new
    features at each stage, so the spatial graph co-evolves with the
    representations.

    Stage pipeline (repeated n_stages times):
        F [T, N, L², C]
        → PearsonAdj(F)   A [T, N, L², L²]    pixel-correlation graph
        → GCN layers on (F, A)                 correlation-driven message pass
        → SpikingResBlock(F)                   spiking feature refinement

    Output: K-hop centre aggregation → [T, N, C]
    """

    def __init__(
        self,
        hidden: int,
        n_stages: int,
        gcn_layers: int,
        K_hop: int,
    ) -> None:
        super().__init__()
        # Per-stage: gcn_layers graph convolutions + SpikingResBlock
        self.stage_gcns = nn.ModuleList([
            nn.ModuleList([GraphConvLayer(hidden, hidden) for _ in range(gcn_layers)])
            for _ in range(n_stages)
        ])
        self.stage_bns = nn.ModuleList([
            nn.ModuleList([layer.BatchNorm1d(hidden) for _ in range(gcn_layers)])
            for _ in range(n_stages)
        ])
        self.stage_lifs = nn.ModuleList([
            nn.ModuleList([
                neuron.LIFNode(
                    decay_input=False, detach_reset=True,
                    surrogate_function=surrogate.ATan(),
                )
                for _ in range(gcn_layers)
            ])
            for _ in range(n_stages)
        ])
        self.stage_res  = nn.ModuleList([SpikingResBlock(hidden) for _ in range(n_stages)])
        self.aggregator = KHopCenterAggregator(K_hop, hidden)
        self.agg_bn     = layer.BatchNorm1d(hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : [T, N, L², C]  →  [T, N, C]"""
        adj = None
        for gcns, bns, lifs, res_block in zip(
                self.stage_gcns, self.stage_bns, self.stage_lifs, self.stage_res):
            # ── Re-estimate spatial correlation graph ────────────────────────
            adj = _pearson_adj(x)                    # [T, N, L², L²]
            # ── Correlation-driven spatial message passing ───────────────────
            for gcn, bn, lif in zip(gcns, bns, lifs):
                res = x
                x   = gcn(x, adj)
                x   = bn(x.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
                x   = x + res                        # residual
                x   = lif(x)
            # ── Spiking feature refinement ───────────────────────────────────
            x = res_block(x)

        T, N, L2, C = x.shape
        center_idx = (L2 - 1) // 2
        agg = self.aggregator(x, adj, center_idx)    # [T, N, C]
        agg = self.agg_bn(
            agg.unsqueeze(-2).permute(0, 1, 3, 2)
        ).squeeze(-1)                                # [T, N, C]
        return agg


# ─────────────────────────────────────────────────────────────────────────────
#  Main model
# ─────────────────────────────────────────────────────────────────────────────

class CorrelationAwareSNN(nn.Module):
    """
    Correlation-Aware SNN for Hyperspectral Image Classification
    =============================================================
    Dual-branch design where correlation IS the processing — not an add-on.

      Spectral branch  — channel-axis correlation:
          CACW gates hidden channels at every stage based on their
          pairwise Pearson correlation across L² spatial pixels.

      Spatial branch   — pixel-axis correlation:
          Dynamic Pearson adjacency re-built at every stage drives GCN
          message passing between the L² patch pixels.

      Cross-branch fusion:
          A 2×2 Pearson correlation matrix between the two branch
          outputs (treating C channels as observations) acts as a
          "uniqueness detector": the more orthogonal the branches,
          the more balanced the softmax blend β.

    Parameters
    ----------
    T         : SNN timesteps.
    img_size  : Spatial patch side-length (patch = img_size × img_size).
    num_cls   : Number of land-cover classes.
    input_dim : Number of hyperspectral input bands.
    hidden    : Hidden feature dimension throughout.
    n_stages  : Encoder stages per branch.
    gcn_layers: GCN layers per stage in the spatial branch.
    K_hop     : K-hop radius for the spatial branch centre aggregator.
    cacw_ratio: CACW MLP width = max(4, int(ratio × hidden)).
    use_cupy  : Enable CuPy spiking backend.
    """

    def __init__(
        self,
        T: int            = 3,
        img_size: int     = 15,
        num_cls: int      = 9,
        input_dim: int    = 103,
        hidden: int       = 32,
        n_stages: int     = 4,
        gcn_layers: int   = 2,
        K_hop: int        = 2,
        cacw_ratio: float = 0.8,
        use_cupy: bool    = False,
    ) -> None:
        super().__init__()
        self.T        = T
        self.img_size = img_size
        self.hidden   = hidden
        self.L2       = img_size * img_size

        # ── Shared spectral projection ───────────────────────────────────────
        # Both branches start from the same C-dimensional spiking embeddings.
        self.proj_fc1  = layer.Linear(input_dim, hidden * 2)
        self.proj_bn1  = layer.BatchNorm1d(hidden * 2)
        self.proj_lif1 = neuron.LIFNode(
            decay_input=False, detach_reset=True,
            surrogate_function=surrogate.ATan())
        self.proj_fc2  = layer.Linear(hidden * 2, hidden)
        self.proj_bn2  = layer.BatchNorm1d(hidden)
        self.proj_lif2 = neuron.LIFNode(
            decay_input=False, detach_reset=True,
            surrogate_function=surrogate.ATan())

        # ── Two branches ────────────────────────────────────────────────────
        self.spectral_branch = SpectralBranch(hidden, n_stages, cacw_ratio)
        self.spatial_branch  = SpatialBranch(hidden, n_stages, gcn_layers, K_hop)

        # ── Cross-branch fusion via CACW (n=2 branches, m=C observations) ───
        # The 2×2 correlation matrix captures how redundant / complementary
        # the spectral and spatial branches are.  MLP maps each branch's row
        # to a scalar; softmax normalises → blend weights β ∈ ℝ².
        self.fusion_cacw = CACW(n=2, d=max(4, int(cacw_ratio * 2)))

        # ── Temporal weighting ───────────────────────────────────────────────
        self.time_logits = nn.Parameter(torch.zeros(T))

        # ── Classifier ───────────────────────────────────────────────────────
        self.classifier = nn.Linear(hidden, num_cls)

        # ── SNN configuration ────────────────────────────────────────────────
        functional.set_step_mode(self, step_mode='m')
        if use_cupy:
            functional.set_backend(self, backend='cupy')

        self._init_weights()

    # ──────────────────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor[N, L², S]

        Returns
        -------
        logits : Tensor[N, num_cls]
        """
        functional.reset_net(self)

        # ── Temporal expand ──────────────────────────────────────────────────
        x = x.unsqueeze(0).expand(self.T, -1, -1, -1)    # [T, N, L², S]

        # ── Shared spectral projection → [T, N, L², C] ──────────────────────
        x = self.proj_fc1(x)
        x = self.proj_bn1(x.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        x = self.proj_lif1(x)
        x = self.proj_fc2(x)
        x = self.proj_bn2(x.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        x = self.proj_lif2(x)                             # [T, N, L², C]

        # ── Two branches (independent processing) ───────────────────────────
        f_spec = self.spectral_branch(x)                  # [T, N, C]
        f_spat = self.spatial_branch(x)                   # [T, N, C]

        # ── Cross-branch correlation fusion ─────────────────────────────────
        # Stack branches as 2 "features", C channels as "observations".
        # CACW(n=2): builds 2×2 Pearson corr → MLP → β ∈ ℝ² per (t, n).
        T, N, C = f_spec.shape
        branches = torch.stack([f_spec, f_spat], dim=-1)  # [T, N, C, 2]
        beta = F.softmax(
            self.fusion_cacw(
                branches.reshape(T * N, C, 2)             # [T·N, C, 2] m=C n=2
            ).reshape(T, N, 2),
            dim=-1,
        )                                                  # [T, N, 2]
        fused = (beta[:, :, 0:1] * f_spec
               + beta[:, :, 1:2] * f_spat)               # [T, N, C]

        # ── Temporal softmax weighting ───────────────────────────────────────
        alpha = torch.softmax(self.time_logits, dim=0)    # [T]
        out   = (alpha.view(-1, 1, 1) * fused).sum(dim=0) # [N, C]

        return self.classifier(out)                        # [N, num_cls]

    # ──────────────────────────────────────────────────────────────────────────

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Linear, layer.Linear)):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.Conv2d, layer.Conv2d)):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (
                layer.BatchNorm1d, layer.BatchNorm2d,
                nn.BatchNorm1d,    nn.BatchNorm2d,
            )):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
