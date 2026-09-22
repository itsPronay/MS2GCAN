"""
Correlation-Aware Spiking Neural Network for Hyperspectral Image Classification
================================================================================
CA-SNN-HSI

Architecture
------------
Correlation awareness is built into the model at three complementary levels:

  Level 1 – Channel Correlation (intra-stage)
      After every spiking encoder stage, a Correlation-Aware Covariance
      Weighting (CACW) module computes the C×C normalized covariance
      (correlation) matrix of the C spectral/hidden channels – treating
      the L² spatial pixels as observations.  An MLP maps each row of
      that matrix to a scalar weight, producing α ∈ ℝ^C.  Channels that
      co-fire heavily (high covariance ↔ redundancy) receive lower
      weights; uniquely firing channels (low covariance ↔ heterogeneity)
      are amplified.  This is the Intra-Stage Weighting (ISW) step.

  Level 2 – Stage Correlation (cross-stage)
      After all K encoder stages, the spatially-pooled activations of
      every stage are stacked and a second CACW module computes the K×K
      correlation between stages, outputting β ∈ ℝ^K.  A softmax over β
      blends the K ISW-adjusted stage outputs into a single representation.
      This is the Cross-Stage Fusion (CSF) step.

  Level 3 – Spatial Correlation (graph)
      An adaptive Pearson-correlation adjacency matrix is built from the
      fused representation and fed to stacked GCN layers, so that
      spatially correlated pixels can share information.

Pipeline
--------
  Input [N, L², S]
    ↓  temporal expand
  [T, N, L², S]
    ↓  Spectral Projection  (FC→BN→LIF × 2)
  [T, N, L², C]
    ↓  Spiking Encoder Stages × K  +  ISW (Level 1)
  {F_1,…,F_K}, {F̃_1,…,F̃_K}
    ↓  Cross-Stage Fusion  (Level 2)
  [T, N, L², C]
    ↓  Pearson GCN  (Level 3)
  [T, N, L², C]
    ↓  K-Hop Centre Aggregation
  [T, N, C]
    ↓  Temporal Softmax Weighting
  [N, C]
    ↓  Linear Classifier
  [N, num_cls]

References
----------
Correlation-Aware Covariance Weighting (CACW) is adapted from:
  "A General Adaptive Dual-level Weighting Mechanism for Remote Sensing
   Pansharpening", CVPR 2025.
  Here it is re-derived for spiking binary features where covariance
  captures inter-channel co-firing patterns rather than pixel statistics.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import neuron, functional, surrogate, layer


# ──────────────────────────────────────────────────────────────────────────────
#  Core primitive: Correlation-Aware Covariance Weighting (CACW)
# ──────────────────────────────────────────────────────────────────────────────

class CACW(nn.Module):
    """
    Correlation-Aware Covariance Weighting.

    Maps a batch of observations X ∈ ℝ^{…×m×n} to per-feature
    importance weights γ ∈ ℝ^{…×n}.

    Steps
    -----
    1. Mean-centre X along the sample axis (dim -2).
    2. Compute the n×n covariance matrix  C = X_c^T X_c / (m-1).
    3. Normalise to a correlation matrix  C̃_ij = C_ij / (‖X̄_i‖·‖X̄_j‖).
    4. Apply a row-wise two-layer MLP to C̃:
           row_i  →  Linear(n, d) → LeakyReLU → Linear(d, 1)
       yielding γ ∈ ℝ^n  (one weight per feature column).

    The correlation matrix encodes redundancy (high |C̃_ij|) and
    heterogeneity (low |C̃_ij|) between feature pairs.  The MLP learns
    to suppress redundant features and amplify heterogeneous ones.

    Args
    ----
    n : int  – number of features to weight.
    d : int  – MLP hidden width  (paper recommends ≈ 0.8 × n).
    """

    def __init__(self, n: int, d: int) -> None:
        super().__init__()
        self.n = n
        self.fc1 = nn.Linear(n, d, bias=True)
        self.act  = nn.LeakyReLU(negative_slope=0.1, inplace=False)
        self.fc2  = nn.Linear(d, 1, bias=False)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        X : Tensor[..., m, n]  –  m samples × n features.

        Returns
        -------
        gamma : Tensor[..., n]
        """
        # ── 1. Mean-centre ──────────────────────────────────────────────────
        Xc = X - X.mean(dim=-2, keepdim=True)                # [..., m, n]

        # ── 2. Pearson correlation matrix C̃ ∈ [-1, 1]^{n×n} ───────────────
        #   C̃_ij = (Xc[:,i] · Xc[:,j]) / (‖Xc[:,i]‖ · ‖Xc[:,j]‖)
        #   This equals the standard Pearson r; the (m-1) factors cancel.
        #   Normalise each column (feature) of Xc to unit L2 norm:
        Xc_normed = F.normalize(Xc, p=2, dim=-2, eps=1e-8)   # [..., m, n]
        C_tilde   = Xc_normed.transpose(-2, -1) @ Xc_normed  # [..., n, n]
        # diagonal = 1 exactly; off-diagonal ∈ (-1, 1)

        # ── 3. Row-wise MLP ─────────────────────────────────────────────────
        #   Row i of C_tilde = [r(i,0), r(i,1), …, r(i,n-1)]
        #   → encodes how feature i correlates with every other feature.
        h     = self.act(self.fc1(C_tilde))   # [..., n, d]
        gamma = self.fc2(h).squeeze(-1)        # [..., n]
        return gamma


# ──────────────────────────────────────────────────────────────────────────────
#  Level-1 module: Intra-Stage Channel Weighting  (ISW)
# ──────────────────────────────────────────────────────────────────────────────

class IntraStageWeighting(nn.Module):
    """
    Level-1 correlation weighting: channels within one encoder stage.

    Given spiking features F ∈ ℝ^{T×N×L²×C}:
      • Role of "samples"  → L² spatial pixels   (m = L²).
      • Role of "features" → C hidden channels    (n = C).
      • T and N are merged so each (timestep, sample) pair contributes
        its own spatial covariance matrix, giving CACW more statistics.
      • CACW yields α ∈ ℝ^{T·N×C} → reshaped to [T, N, 1, C].
      • F̃ = F ⊙ α  (element-wise channel scaling).

    In the spiking context, C̃_ij captures the co-firing rate between
    channel i and channel j over the L² spatial positions, making the
    correlation matrix a direct measure of spike-level redundancy.

    Args
    ----
    C : int – channel (hidden) dimension.
    d : int – CACW MLP hidden width.
    """

    def __init__(self, C: int, d: int) -> None:
        super().__init__()
        self.cacw = CACW(n=C, d=d)

    def forward(self, F: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        F : Tensor[T, N, L², C]

        Returns
        -------
        F_tilde : Tensor[T, N, L², C]
        """
        T, N, L2, C = F.shape
        X     = F.reshape(T * N, L2, C)             # [T·N, L², C]  m=L², n=C
        alpha = torch.sigmoid(self.cacw(X))          # [T·N, C] ∈ (0,1)
        # Sigmoid bounds weights to (0,1): 0 = fully suppress, 1 = pass through.
        # Unbounded raw logits would flip signs of spike-residual features,
        # destabilising subsequent LIF neurons.
        alpha = alpha.reshape(T, N, 1, C)            # broadcast over L²
        return F * alpha                              # [T, N, L², C]


# ──────────────────────────────────────────────────────────────────────────────
#  Level-2 module: Cross-Stage Correlation Fusion  (CSF)
# ──────────────────────────────────────────────────────────────────────────────

class CrossStageFusion(nn.Module):
    """
    Level-2 correlation weighting: across K encoder stages.

    Given original stage outputs F_1…F_K and ISW-adjusted outputs F̃_1…F̃_K:
      • Spatially avg-pool each F_k → [T, N, C].
      • Stack → [T, N, K, C], then reshape to [T·N, C, K].
      • Role of "samples"  → C channels   (m = C).
      • Role of "features" → K stages     (n = K).
      • CACW yields β ∈ ℝ^{T·N×K} → softmax-normalised.
      • F̂ = Σ_k  softmax(β)_k · F̃_k

    This mirrors PCA's projection F̂ = β^T F̃ but with data-driven,
    task-specific β instead of a fixed orthogonal basis.

    Args
    ----
    n_stages : int – number of encoder stages K.
    d        : int – CACW MLP hidden width.
    """

    def __init__(self, n_stages: int, d: int) -> None:
        super().__init__()
        self.n_stages = n_stages
        self.cacw = CACW(n=n_stages, d=d)

    def forward(
        self,
        F_list: list[torch.Tensor],
        F_tilde_list: list[torch.Tensor],
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        F_list       : list of K tensors, each [T, N, L², C]
        F_tilde_list : list of K tensors, each [T, N, L², C]

        Returns
        -------
        F_hat : Tensor[T, N, L², C]
        """
        T, N, L2, C = F_list[0].shape
        K = len(F_list)

        # ── Spatially pool each stage ────────────────────────────────────────
        pooled = torch.stack(
            [f.mean(dim=-2) for f in F_list], dim=2
        )  # [T, N, K, C]

        # ── CACW: treat C as samples, K stages as features ───────────────────
        X    = pooled.reshape(T * N, K, C).transpose(-2, -1)  # [T·N, C, K]
        beta = self.cacw(X)                                     # [T·N, K]
        beta = F.softmax(beta.reshape(T, N, K), dim=-1)         # [T, N, K]

        # ── Weighted sum of ISW-adjusted stage outputs ───────────────────────
        F_tilde_stack = torch.stack(F_tilde_list, dim=2)        # [T, N, K, L², C]
        beta_w        = beta.unsqueeze(-1).unsqueeze(-1)         # [T, N, K,  1,  1]
        return (beta_w * F_tilde_stack).sum(dim=2)               # [T, N, L², C]


# ──────────────────────────────────────────────────────────────────────────────
#  Auxiliary spiking layers
# ──────────────────────────────────────────────────────────────────────────────

class SpikingResBlock(nn.Module):
    """
    Residual spiking feature extraction block.

    FC → BN → LIF with an additive skip connection.
    Operates on [T, N, L², C]; BN is applied with axes transposed so
    spikingjelly's multi-step BatchNorm1d sees [T, N, C, L²].
    """

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
        out = self.lif(out)
        return out + res


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

    Computes A^k masks (k = 0 … K) for the centre pixel, gathers
    per-hop feature vectors by weighted pooling, then fuses them via a
    learnable SiLU-gated softmax.
    """

    def __init__(
        self,
        K: int = 2,
        hidden: int = 64,
        include_self: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.K            = K
        self.eps          = eps
        self.include_self = include_self
        self.linear       = layer.Linear(hidden, hidden)
        self.gate = nn.Sequential(
            nn.LayerNorm(hidden * (K + 1)),
            layer.Linear(hidden * (K + 1), hidden),
            nn.SiLU(),
            layer.Linear(hidden, K + 1),
            nn.Softmax(dim=-1),
        )

    def forward(
        self,
        gcn_out: torch.Tensor,
        adj: torch.Tensor,
        center_idx: int,
    ) -> torch.Tensor:
        gcn_out = self.linear(gcn_out)

        if self.K == 0:
            return gcn_out[:, :, center_idx, :]

        T, N, L2, H = gcn_out.shape
        device = adj.device

        adj_flat   = adj.reshape(T * N, L2, L2)
        eye        = (torch.eye(L2, device=device)
                      .unsqueeze(0).expand(T * N, -1, -1))
        adj_powers = [eye.clone()]
        cur        = adj_flat.clone()
        for k in range(1, self.K + 1):
            adj_powers.append(cur)
            if k < self.K:
                cur = torch.bmm(cur, adj_flat)

        gcn_flat = gcn_out.reshape(T * N, L2, H)
        feats: list[torch.Tensor] = []
        for k, ap in enumerate(adj_powers):
            mask = ap[:, center_idx, :]                           # [B, L²]
            if not self.include_self and k > 0:
                zm   = torch.zeros_like(mask)
                keep = torch.arange(L2, device=device) != center_idx
                zm[:, keep] = mask[:, keep]
                mask = zm
            mask = mask / (mask.sum(-1, keepdim=True) + self.eps)
            feats.append(
                torch.bmm(mask.unsqueeze(1), gcn_flat).squeeze(1)
            )

        feat_cat   = torch.cat(feats,   dim=-1)   # [B, H·(K+1)]
        feat_stack = torch.stack(feats, dim=-1)   # [B, H, K+1]
        weights    = self.gate(feat_cat).unsqueeze(-2)  # [B, 1, K+1]
        fused      = (weights * feat_stack).sum(-1)     # [B, H]
        return fused.view(T, N, H)


# ──────────────────────────────────────────────────────────────────────────────
#  Main model
# ──────────────────────────────────────────────────────────────────────────────

class CorrelationAwareSNN(nn.Module):
    """
    Correlation-Aware Spiking Neural Network for Hyperspectral Image Classification
    ================================================================================

    A fully integrated model where correlation awareness operates at
    three nested levels:

      • **Channel level**  (IntraStageWeighting / ISW):
            Co-firing spectral channels are identified via their pairwise
            Pearson correlation and down-weighted, reducing spectral
            redundancy within every encoder stage.

      • **Stage level**    (CrossStageFusion / CSF):
            Encoder stages whose outputs are highly correlated with each
            other contribute less to the fused representation, while
            stages capturing unique information are amplified.

      • **Spatial level**  (Pearson GCN):
            Pixels with similar spike-firing profiles are connected in a
            dynamic graph, enabling correlation-aware spatial message
            passing via GCN.

    Parameters
    ----------
    T         : int   – SNN time steps.
    img_size  : int   – Side length of the square spatial patch.
    num_cls   : int   – Number of land-cover classes.
    input_dim : int   – Number of hyperspectral bands.
    hidden    : int   – Hidden feature dimension throughout the network.
    n_stages  : int   – Number of sequential spiking encoder stages.
    gcn_layers: int   – Depth of the Pearson GCN.
    K_hop     : int   – K-hop radius for the centre aggregator.
    cacw_ratio: float – ISW CACW MLP width = max(4, int(ratio × hidden)).
    use_cupy  : bool  – Use CuPy backend for SNN layers.
    """

    def __init__(
        self,
        T: int          = 3,
        img_size: int   = 15,
        num_cls: int    = 9,
        input_dim: int  = 103,
        hidden: int     = 32,
        n_stages: int   = 4,
        gcn_layers: int = 2,
        K_hop: int      = 2,
        cacw_ratio: float = 0.8,
        use_cupy: bool  = False,
    ) -> None:
        super().__init__()
        self.T        = T
        self.img_size = img_size
        self.hidden   = hidden
        self.n_stages = n_stages
        self.L2       = img_size * img_size

        # ── Spectral projection ──────────────────────────────────────────────
        self.fc1      = layer.Linear(input_dim, hidden * 2)
        self.fc1_bn   = layer.BatchNorm1d(hidden * 2)
        self.fc1_lif  = neuron.LIFNode(
            decay_input=False, detach_reset=True,
            surrogate_function=surrogate.ATan())
        self.fc2      = layer.Linear(hidden * 2, hidden)
        self.fc2_bn   = layer.BatchNorm1d(hidden)
        self.fc2_lif  = neuron.LIFNode(
            decay_input=False, detach_reset=True,
            surrogate_function=surrogate.ATan())

        # ── Spiking encoder stages ───────────────────────────────────────────
        self.encoder_stages = nn.ModuleList(
            [SpikingResBlock(hidden) for _ in range(n_stages)]
        )

        # ── Level-1: Intra-Stage Channel Weighting (ISW) ────────────────────
        isw_d = max(4, int(cacw_ratio * hidden))
        self.isw_modules = nn.ModuleList(
            [IntraStageWeighting(C=hidden, d=isw_d) for _ in range(n_stages)]
        )

        # ── Level-2: Cross-Stage Correlation Fusion (CSF) ───────────────────
        csf_d = max(4, int(cacw_ratio * n_stages))
        self.csf = CrossStageFusion(n_stages=n_stages, d=csf_d)

        # ── Level-3: Pearson GCN ─────────────────────────────────────────────
        self.gcn_list = nn.ModuleList(
            [GraphConvLayer(hidden, hidden) for _ in range(gcn_layers)]
        )
        self.gcn_bns = nn.ModuleList(
            [layer.BatchNorm1d(hidden) for _ in range(gcn_layers)]
        )
        self.gcn_lifs = nn.ModuleList([
            neuron.LIFNode(
                decay_input=False, detach_reset=True,
                surrogate_function=surrogate.ATan())
            for _ in range(gcn_layers)
        ])

        # ── K-hop centre aggregation ─────────────────────────────────────────
        self.aggregator = KHopCenterAggregator(K_hop, hidden, include_self=True)
        self.fusion_bn  = layer.BatchNorm1d(hidden)

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
        Forward pass.

        Parameters
        ----------
        x : Tensor[N, L², S]  –  batch of HSI spatial patches.

        Returns
        -------
        logits : Tensor[N, num_cls]
        """
        functional.reset_net(self)

        # Expand across T time steps.  LIF neurons are deterministic;
        # temporal diversity emerges from membrane-potential accumulation
        # across timesteps (each step's threshold crossings depend on the
        # accumulated potential from all previous steps).
        x = x.unsqueeze(0).expand(self.T, -1, -1, -1)   # [T, N, L², S]

        # ── Spectral projection ──────────────────────────────────────────────
        x = self.fc1(x)
        x = self.fc1_bn(x.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        x = self.fc1_lif(x)
        x = self.fc2(x)
        x = self.fc2_bn(x.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        x = self.fc2_lif(x)                              # [T, N, L², C]

        # ── Spiking encoder + Level-1 (ISW) ─────────────────────────────────
        F_orig: list[torch.Tensor]    = []
        F_weighted: list[torch.Tensor] = []
        feat = x
        for stage, isw in zip(self.encoder_stages, self.isw_modules):
            feat = stage(feat)             # SpikingResBlock → [T, N, L², C]
            F_orig.append(feat)
            F_weighted.append(isw(feat))   # ISW             → [T, N, L², C]

        # ── Level-2 (CSF) ────────────────────────────────────────────────────
        fused = self.csf(F_orig, F_weighted)              # [T, N, L², C]

        # ── Level-3: adaptive Pearson graph + GCN ────────────────────────────
        adj     = self._pearson_adj(fused)                # [T, N, L², L²]
        gcn_out = fused
        for gcn_layer, gcn_bn, gcn_lif in zip(
                self.gcn_list, self.gcn_bns, self.gcn_lifs):
            res     = gcn_out
            gcn_out = gcn_layer(gcn_out, adj)
            gcn_out = gcn_bn(
                gcn_out.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
            gcn_out = gcn_out + res
            gcn_out = gcn_lif(gcn_out)

        # ── K-hop centre aggregation ─────────────────────────────────────────
        center_idx = (self.L2 - 1) // 2
        agg = self.aggregator(gcn_out, adj, center_idx)  # [T, N, C]
        agg = self.fusion_bn(
            agg.unsqueeze(-2).permute(0, 1, 3, 2)
        ).squeeze(-1)                                     # [T, N, C]

        # ── Temporal weighting ───────────────────────────────────────────────
        alpha  = torch.softmax(self.time_logits, dim=0)   # [T]
        out    = (alpha.view(-1, 1, 1) * agg).sum(dim=0)  # [N, C]

        return self.classifier(out)                        # [N, num_cls]

    # ──────────────────────────────────────────────────────────────────────────

    def _pearson_adj(self, spike: torch.Tensor) -> torch.Tensor:
        """
        Build a symmetric, degree-normalised Pearson similarity graph.

        Pixels whose spike patterns (across hidden channels) are
        positively correlated are connected by a strong edge, enabling
        GCN to propagate information between spectrally similar spatial
        neighbours.
        """
        mu       = spike.mean(dim=-1, keepdim=True)
        centered = spike - mu
        # eps=1e-8 guards zero-variance pixels (e.g. zero-padded patch borders)
        normed   = F.normalize(centered, p=2, dim=-1, eps=1e-8)
        sim      = torch.matmul(normed, normed.transpose(-2, -1))
        adj      = F.relu(sim)
        deg      = adj.sum(-1, keepdim=True)
        d_inv    = torch.pow(deg + 1e-6, -0.5)
        return adj * d_inv * d_inv.transpose(-2, -1)

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
