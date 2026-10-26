"""DSR-SAM2: dual-branch SAM with CKA cross-branch attention.

Architecture (input 256x256 -> ViT 16x16 tokens):

  ViT branch (unchanged SAM ViT-B):
      patch embed 16x16 -> 12 transformer blocks (window/global attention)
      -> final ViT/CNN fusion -> SAM neck (256ch).

  CNN branch (ResNet18, IMAGENET1K_V1):
      stem/layer1/layer2/layer3 -> 16x16x256
      -> layer4 (new 256->768 ResBlock, 16x16x768), the 7-ResBlock base
      -> layer5..layer(4+cba_last_n) (new 768->768 ResBlocks, 16x16x768).

  Cross-branch attention (2*cba_last_n CBA blocks):
      The last cba_last_n ViT blocks pair one-to-one with the additional
      CNN ResBlocks (ViT block depth-n+r with layer 5+r).
      Each ViT block has two CBA blocks:
        * after norm1, with the resblock "before" feature;
        * after norm2, with the resblock "after" feature.
      Each CBA computes the per-channel linear CKA (Centered Kernel
      Alignment) between the same-index spatial channels:
          C_weight[c] = CKA(f_vit[..., c], f_cnn[..., c])
                      = <v_c, n_c>^2 / (||v_c||^2 ||n_c||^2)
      where v_c and n_c are the H*W spatial vectors centered over space, so
      C_weight lies in [0, 1] (the squared Pearson correlation).  An
      AttentionGate (Oktay et al. 2018) uses f_vit as the gating signal to
      filter f_cnn:
          f_gated_cnn = AttentionGate(f_cnn, f_vit)
      and the ViT features are expanded by
          f_vit + ((1 - C_weight) * f_cnn + f_gated_cnn) / 2
      so the CNN branch contributes most when the two branches are least
      aligned, while the gate highlights the relevant CNN regions.
      Off-diagonal channel pairs are not used.
      Then BUSSAM's spatial-attention + MLP adapter runs.

  Final ViT/CNN fusion:
      x = x + cnn_final
      (plain addition, no CKA and no attention gate).

  ``use_attention_gate=False`` removes the CBA gates and falls back to
  ``f_vit + (1 - C_weight) * f_cnn``.  ``cka_fusion=False`` bypasses the
  CBA gate and CKA weighting for its ablation.  CKA and the CBA gates are
  always differentiable.

  Everything else (LoRA/EMA, prompt encoder, mask decoder) is
  inherited from DSR-SAM.
"""
import copy
import math
from functools import partial
from typing import Any, Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # torchvision is only needed for the ImageNet-pretrained ResNet18 branch
    from torchvision.models import ResNet18_Weights, resnet18
except ImportError:  # pragma: no cover - torchvision is present in khoa-env
    ResNet18_Weights = None
    resnet18 = None

from utils.seed import deterministic_warn_only


# ============================================================
# Common building blocks (repo segment_anything/modeling/common.py)
# ============================================================

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class MLPBlock(nn.Module):
    def __init__(self, embedding_dim: int, mlp_dim: int,
                 act: Type[nn.Module] = nn.GELU) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


# ============================================================
# Image encoder (repo segment_anything/modeling/image_encoder.py,
# with the ema flag threaded through to the LoRA qkv modules)
# ============================================================

def window_partition(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    B, H, W, C = x.shape
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w
    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows, (Hp, Wp)


def window_unpartition(windows: torch.Tensor, window_size: int,
                       pad_hw: Tuple[int, int], hw: Tuple[int, int]) -> torch.Tensor:
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)
    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x


def get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    if rel_pos.shape[0] != max_rel_dist:
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist, mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)
    return rel_pos_resized[relative_coords.long()]


def add_decomposed_rel_pos(attn: torch.Tensor, q: torch.Tensor,
                           rel_pos_h: torch.Tensor, rel_pos_w: torch.Tensor,
                           q_size: Tuple[int, int], k_size: Tuple[int, int]) -> torch.Tensor:
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)
    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)
    attn = (attn.view(B, q_h, q_w, k_h, k_w) + rel_h[:, :, :, :, None]
            + rel_w[:, :, :, None, :]).view(B, q_h * q_w, k_h * k_w)
    return attn


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = True,
                 use_rel_pos: bool = False, rel_pos_zero_init: bool = True,
                 input_size: Optional[Tuple[int, int]] = None) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert input_size is not None, "Input size must be provided if using relative positional encoding."
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

    def forward(self, x: torch.Tensor, ema: bool = False) -> torch.Tensor:
        B, H, W, _ = x.shape
        qkv = self.qkv(x, ema=ema).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)
        attn = (q * self.scale) @ k.transpose(-2, -1)
        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))
        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = self.proj(x)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, kernel_size: Tuple[int, int] = (16, 16),
                 stride: Tuple[int, int] = (16, 16), padding: Tuple[int, int] = (0, 0),
                 in_chans: int = 3, embed_dim: int = 768) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size,
                              stride=stride, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.permute(0, 2, 3, 1)
        return x


class Block(nn.Module):
    """ViT block with window/global attention (repo image_encoder.py).

    DSR-SAM2 extension: the last ``cba_last_n`` blocks also own two
    Cross-Branch Attention (CBA) modules, inserted exactly where BUSSAM
    inserts its Cross-Branch Adapter:

      * ``cba1`` acts on ``norm1(x)`` before the attention (residual add);
      * ``cba2`` acts on ``norm2(x)`` after the MLP (residual add).

    The ViT/CNN fusion itself is not BUSSAM's direct ``f_vit + f_cnn``: the
    CNN branch is added as
    ``f_vit + ((1 - C_weight) * f_cnn + f_gated_cnn) / 2`` per channel, as
    described in ``CrossBranchAttention``.
    """

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 qkv_bias: bool = True, norm_layer: Type[nn.Module] = nn.LayerNorm,
                 act_layer: Type[nn.Module] = nn.GELU, use_rel_pos: bool = False,
                 rel_pos_zero_init: bool = True, window_size: int = 0,
                 input_size: Optional[Tuple[int, int]] = None,
                 use_cba: bool = False,
                 use_attention_gate: bool = True,
                 attention_gate_channels: Optional[int] = None,
                 cka_fusion: bool = True) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                              use_rel_pos=use_rel_pos, rel_pos_zero_init=rel_pos_zero_init,
                              input_size=input_size if window_size == 0 else (window_size, window_size))
        self.norm2 = norm_layer(dim)
        self.mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer)
        self.window_size = window_size
        self.use_cba = use_cba
        if use_cba:
            # BUSSAM's Cross-Branch Adapter is an MLPAdapter with
            # skip_connect=False; the residual is applied by the block.
            self.cba1 = CrossBranchAttention(
                dim, mlp_ratio=0.25, act_layer=act_layer, skip_connect=False,
                use_attention_gate=use_attention_gate,
                attention_gate_channels=attention_gate_channels,
                fusion=cka_fusion)
            self.cba2 = CrossBranchAttention(
                dim, mlp_ratio=0.25, act_layer=act_layer, skip_connect=False,
                use_attention_gate=use_attention_gate,
                attention_gate_channels=attention_gate_channels,
                fusion=cka_fusion)

    def forward(self, x: torch.Tensor, ema: bool = False,
                cnn_before: Optional[torch.Tensor] = None,
                cnn_after: Optional[torch.Tensor] = None) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)
        if self.use_cba and cnn_before is not None:
            # CBA n.1 (of the 2 in this block) + residual, before attention.
            x = x + self.cba1(x, cnn_before)
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)
        x = self.attn(x, ema=ema)
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))
        x = shortcut + x
        xn = self.norm2(x)
        x = x + self.mlp(xn)
        if self.use_cba and cnn_after is not None:
            # CBA n.2 of this block + residual, after the MLP.
            x = x + self.cba2(xn, cnn_after)
        return x


# ============================================================
# DSR-SAM2 CNN branch, per-channel CKA and CBA (BUSSAM cross-branch
# attention, ViT features expanded by the CKA disagreement weight)
# ============================================================

class SpatialAttention(nn.Module):
    """BUSSAM spatial attention (repo common.py): max+avg -> 7x7 conv -> sigmoid."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7,
                              padding=7 // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        out = torch.cat([max_out, avg_out], dim=1)
        out = self.sigmoid(self.conv(out))
        return x * out


class BasicResBlock(nn.Module):
    """ResNet18 BasicBlock (two 3x3 convs + identity/projection shortcut)."""

    def __init__(self, in_chans: int, out_chans: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_chans, out_chans, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_chans)
        self.conv2 = nn.Conv2d(out_chans, out_chans, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_chans)
        self.relu = nn.ReLU(inplace=True)
        if stride != 1 or in_chans != out_chans:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_chans, out_chans, kernel_size=1, stride=stride,
                          bias=False),
                nn.BatchNorm2d(out_chans),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class ResNet18Branch(nn.Module):
    """CNN branch: ImageNet1k-v1 ResNet18 up to layer3 (16x16x256), then
    one 256->768 ResBlock (layer4) and ``num_extra_blocks`` additional
    768->768 ResBlocks (layer5 ... layer(4+num_extra_blocks)), all at 16x16
    for a 256x256 input.

    The fixed base is 7 ResBlocks: layer1-3 (2 each, pretrained) + layer4
    (projection, new).  The additional ResBlocks are new and are consumed by
    the last ``num_extra_blocks`` ViT blocks: each block's *before* feature
    feeds CBA n.1 and its *after* feature feeds CBA n.2.
    """

    def __init__(self, pretrained: bool = True, out_chans: int = 768,
                 num_extra_blocks: int = 4) -> None:
        super().__init__()
        if resnet18 is None:
            raise ImportError('torchvision is required for the ResNet18 CNN branch')
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet18(weights=weights)
        # Everything up to layer3 keeps the pretrained ImageNet1k-v1 weights.
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1,
                                  backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1   # 64ch,  64x64
        self.layer2 = backbone.layer2   # 128ch, 32x32
        self.layer3 = backbone.layer3   # 256ch, 16x16
        # Fixed base projection (new, no pretrained weights).
        self.layer4 = BasicResBlock(256, out_chans, stride=1)   # 16x16x768
        # Additional branch-specific layers (new, no pretrained weights).
        # Named layer5, layer6, ... to keep n=4 state-dict compatibility.
        self.num_extra_blocks = max(0, int(num_extra_blocks))
        for i in range(self.num_extra_blocks):
            setattr(self, f'layer{5 + i}', BasicResBlock(out_chans, out_chans))
        self.out_chans = out_chans

    def forward(self, x: torch.Tensor):
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        pairs = []
        for i in range(self.num_extra_blocks):
            layer = getattr(self, f'layer{5 + i}')
            before = x
            x = layer(x)
            pairs.append((before, x))
        return pairs, x


def _align_spatial(f_vit: torch.Tensor, f_cnn: torch.Tensor) -> torch.Tensor:
    """Resize ``f_cnn`` to the ViT spatial grid when the two grids differ."""
    if f_vit.shape[1:3] != f_cnn.shape[1:3]:
        f_cnn = F.interpolate(f_cnn.permute(0, 3, 1, 2),
                              size=f_vit.shape[1:3], mode='bilinear',
                              align_corners=False).permute(0, 2, 3, 1)
    return f_cnn


def per_channel_cka_score(f_vit: torch.Tensor, f_cnn: torch.Tensor,
                          eps: float = 1e-12) -> torch.Tensor:
    """Per-channel linear CKA ``C_weight`` between ViT and CNN features.

    ``f_vit`` and ``f_cnn`` are ``(B, H, W, C)`` (here ``B, 16, 16, 768``).
    For each same-index channel pair, the ``H*W`` spatial values are centered
    over space and compared with the linear CKA, which for two 1-D variables
    is exactly the squared Pearson correlation:

        C_weight[c] = <v_c, n_c>^2 / (||v_c||^2 ||n_c||^2)  in [0, 1]

    ``C_weight = 1`` means the two channels are perfectly aligned, ``0``
    means they are uncorrelated.  Only same-index channel pairs are used.
    Returns ``(B, C)``.  The score is always differentiable.
    """
    if f_vit.shape[1:3] != f_cnn.shape[1:3]:
        # Defensive: the CNN and ViT maps must have the same spatial grid.
        f_cnn = _align_spatial(f_vit, f_cnn)
    # CKA is evaluated in fp32 even if the branches run under autocast.
    f_vit = f_vit.float()
    f_cnn = f_cnn.float()
    b, h, w, c = f_vit.shape
    v = f_vit.reshape(b, h * w, c)
    n = f_cnn.reshape(b, h * w, c)
    # Center each channel over its H*W spatial positions.
    v = v - v.mean(dim=1, keepdim=True)
    n = n - n.mean(dim=1, keepdim=True)
    cross = (v * n).sum(dim=1)                            # (B, C)
    var_v = (v * v).sum(dim=1)                            # (B, C)
    var_n = (n * n).sum(dim=1)                            # (B, C)
    c_weight = (cross * cross) / (var_v * var_n).clamp_min(eps)
    return c_weight.clamp(0.0, 1.0)


def cka_fused_features(f_vit: torch.Tensor, f_cnn: torch.Tensor,
                       gate: Optional[nn.Module] = None,
                       use_weight: bool = True,
                       return_weight: bool = False):
    """Fuse ``f_vit``/``f_cnn`` with CKA and an optional attention gate.

    With ``gate`` (paper-style additive attention gate) and
    ``use_weight=True``:

        f_gated_cnn = gate(f_cnn, f_vit)          # f_vit gates f_cnn
        fused = f_vit + ((1 - C_weight) * f_cnn + f_gated_cnn) / 2

    Without a gate, the legacy fusion is used:

        fused = f_vit + (1 - C_weight) * f_cnn

    ``use_weight=False`` returns the unweighted ``f_vit + f_cnn`` and bypasses
    the gate (the CKA term is still computed when ``return_weight=True`` so
    ``L_CKA`` remains available for the ablation).

    ``return_weight=True`` also returns ``C_weight``; the CBA blocks use this
    to collect the per-channel terms for ``L_CKA``.
    """
    if not use_weight and not return_weight:
        return f_vit + f_cnn
    # Align once so the CKA score, the gate, and the fused residual share a grid.
    f_cnn = _align_spatial(f_vit, f_cnn)
    c_weight = per_channel_cka_score(f_vit, f_cnn)
    if use_weight:
        w = (1.0 - c_weight)[:, None, None, :]   # (B, 1, 1, C)
        cnn_term = w * f_cnn
        if gate is not None:
            f_gated_cnn = gate(f_cnn, f_vit)
            fused = f_vit + (cnn_term + f_gated_cnn) / 2.0
        else:
            fused = f_vit + cnn_term
    else:
        fused = f_vit + f_cnn
    if return_weight:
        return fused, c_weight
    return fused


class AttentionGate(nn.Module):
    """Additive attention gate from Attention U-Net (Oktay et al. 2018).

    Paper Eq. (1)-(2), with f_vit as the gating signal ``g`` and f_cnn as the
    skip feature ``x``:

        q     = psi( ReLU( W_x(f_cnn) + W_g(f_vit) ) )
        alpha = sigmoid(q)
        f_gated_cnn = f_cnn * alpha

    ``W_x``, ``W_g`` and ``psi`` are 1x1 convolutions with biases; there is
    no BatchNorm in the paper formulation.  ``inter_channels`` is ``F_int``
    (default: ``channels``, i.e. C).
    """

    def __init__(self, channels: int,
                 inter_channels: Optional[int] = None) -> None:
        super().__init__()
        inter = channels if inter_channels is None else int(inter_channels)
        self.W_x = nn.Conv2d(channels, inter, kernel_size=1, bias=True)
        self.W_g = nn.Conv2d(channels, inter, kernel_size=1, bias=True)
        self.psi = nn.Conv2d(inter, 1, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, f_cnn: torch.Tensor,
                f_vit: torch.Tensor) -> torch.Tensor:
        # (B, H, W, C) -> (B, C, H, W) for the 1x1 convolutions.
        x = f_cnn.permute(0, 3, 1, 2)
        g = f_vit.permute(0, 3, 1, 2)
        q = self.psi(self.relu(self.W_x(x) + self.W_g(g)))
        alpha = self.sigmoid(q)
        return (x * alpha).permute(0, 2, 3, 1)


class CrossBranchAttention(nn.Module):
    """CBA block: CKA + attention-gated ViT/CNN expansion + MLP adapter.

    BUSSAM's MLPAdapter starts with ``x = f_vit + f_cnn``.  DSR-SAM2 replaces
    that sum by, for every channel c (with C_weight in [0, 1]),

        f_gated_cnn[c] = AttentionGate(f_cnn, f_vit)
        fused[..., c] = f_vit[..., c]
                        + ((1 - C_weight[c]) * f_cnn[..., c]
                           + f_gated_cnn[c]) / 2

    so the CNN branch expands the ViT representation instead of replacing it.
    C_weight[c] is the per-channel linear CKA between the same-index channels
    of ``f_cnn`` and ``f_vit``.  Other channel pairs are not used.  The block
    then continues with BUSSAM's SpatialAttention + Linear/GELU/Linear.  With
    ``skip_connect=False`` the returned tensor is the adapter output (the
    owning ``Block`` applies the residual, exactly like BUSSAM).

    ``fusion`` selects the f_vit/f_cnn combination:
      * ``True``  (default) -> CKA + attention-gated fusion above
      * ``False`` (ablation) -> f_vit + f_cnn (gate bypassed)

    ``last_cka`` stores the ``(B, C)`` C_weight of the most recent forward so
    the encoder can aggregate ``L_CKA`` across all CBA modules.
    """

    def __init__(self, dim: int, mlp_ratio: float = 0.25,
                 act_layer: Type[nn.Module] = nn.GELU,
                 skip_connect: bool = False,
                 use_attention_gate: bool = True,
                 attention_gate_channels: Optional[int] = None,
                 fusion: bool = True) -> None:
        super().__init__()
        if not isinstance(fusion, bool):
            raise TypeError(f'fusion must be a bool, got {fusion!r}')
        self.fusion = fusion
        self.skip_connect = skip_connect
        self.last_cka = None
        if use_attention_gate and fusion:
            self.attention_gate = AttentionGate(
                dim, inter_channels=attention_gate_channels)
        else:
            # fusion=False bypasses the gate entirely; don't allocate it.
            self.attention_gate = None
        hidden = int(dim * mlp_ratio)
        self.act = act_layer()
        self.D_fc1 = nn.Linear(dim, hidden)
        self.D_fc2 = nn.Linear(hidden, dim)
        self.spatial_attention = SpatialAttention()

    def forward(self, f_vit: torch.Tensor, f_cnn: torch.Tensor) -> torch.Tensor:
        fused, c_weight = cka_fused_features(
            f_vit, f_cnn, gate=self.attention_gate, use_weight=self.fusion,
            return_weight=True)
        # Keep the CKA term for the (optional) L_CKA loss.  Cleared by
        # ImageEncoderViT.cka_loss() after the scalar loss is built.
        self.last_cka = c_weight
        xs = self.spatial_attention(fused.permute(0, 3, 1, 2))
        xs = xs.permute(0, 2, 3, 1)
        xs = self.D_fc2(self.act(self.D_fc1(xs)))
        if self.skip_connect:
            fused = fused + xs
        else:
            fused = xs
        return fused


class ImageEncoderViT(nn.Module):
    """SAM ViT encoder + ResNet18 CNN branch + CBA-enabled last ViT blocks.

    Input goes through both branches in parallel:

      * ViT branch: the original SAM encoder (patch embed, 12 blocks, neck).
      * CNN branch: ResNet18 ImageNet1k-v1 through layer3 (16x16x256), then
        ResBlock layer4 (16x16x768).  The fixed base is 7 ResBlocks.
      * ``cba_last_n = n`` adds ``n`` extra 768->768 ResBlocks (layer5 ...
        layer(4+n)) and enables two CBA blocks in each of the last ``n`` ViT
        blocks.  ViT block ``depth-n+r`` pairs with extra layer ``5+r``:
        layer before -> CBA after norm1, layer after -> CBA after norm2.
        Total = 2*n CBA blocks.
      * Before the SAM neck the branches are fused by plain addition:
        ``x = x + cnn_final`` (no CKA, no gate).

    ``cnn_branch=False`` disables the whole dual-branch path (no ResNet18,
    no CBA, no final CNN fusion): the model is plain SAM + LoRA.  It is used
    by the ablation study in ``DG/sota/run_sota.py``.
    """

    def __init__(self, img_size: int = 1024, patch_size: int = 16, in_chans: int = 3,
                 embed_dim: int = 768, depth: int = 12, num_heads: int = 12,
                 mlp_ratio: float = 4.0, out_chans: int = 256, qkv_bias: bool = True,
                 norm_layer: Type[nn.Module] = nn.LayerNorm, act_layer: Type[nn.Module] = nn.GELU,
                 use_abs_pos: bool = True, use_rel_pos: bool = False,
                 rel_pos_zero_init: bool = True, window_size: int = 0,
                 global_attn_indexes: Tuple[int, ...] = (),
                 cnn_pretrained: bool = True, cba_last_n: int = 4,
                 cnn_branch: bool = True, cka_fusion: bool = True,
                 use_attention_gate: bool = True,
                 use_attention_gate_cba: bool = True,
                 attention_gate_channels: Optional[int] = None) -> None:
        super().__init__()
        self.img_size = img_size
        self.cnn_branch = bool(cnn_branch)
        if not self.cnn_branch:
            # Ablation: plain SAM + LoRA; the CNN path (and CBA) is absent.
            cba_last_n = 0
        if isinstance(cba_last_n, bool) or int(cba_last_n) != cba_last_n:
            raise TypeError(f'cba_last_n must be an int, got {cba_last_n!r}')
        cba_last_n = int(cba_last_n)
        if cba_last_n < 0 or cba_last_n > depth:
            raise ValueError(f'cba_last_n must be in [0, {depth}], got {cba_last_n}')
        self.cba_last_n = cba_last_n
        if self.cnn_branch:
            self.cnn_encoder = ResNet18Branch(
                pretrained=cnn_pretrained, out_chans=embed_dim,
                num_extra_blocks=self.cba_last_n)
        else:
            self.cnn_encoder = None
        self.patch_embed = PatchEmbed(kernel_size=(patch_size, patch_size),
                                      stride=(patch_size, patch_size),
                                      in_chans=in_chans, embed_dim=embed_dim)
        self.pos_embed = None
        if use_abs_pos:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, img_size // patch_size, img_size // patch_size, embed_dim)
            )
        self.use_attention_gate = bool(use_attention_gate)
        self.use_attention_gate_cba = bool(use_attention_gate_cba)
        self.attention_gate_channels = attention_gate_channels
        cba_gate = self.use_attention_gate and self.use_attention_gate_cba
        self.blocks = nn.ModuleList()
        for i in range(depth):
            use_cba = i >= depth - self.cba_last_n
            block = Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                          qkv_bias=qkv_bias, norm_layer=norm_layer, act_layer=act_layer,
                          use_rel_pos=use_rel_pos, rel_pos_zero_init=rel_pos_zero_init,
                          window_size=window_size if i not in global_attn_indexes else 0,
                          input_size=(img_size // patch_size, img_size // patch_size),
                          use_cba=use_cba,
                          use_attention_gate=cba_gate,
                          attention_gate_channels=self.attention_gate_channels,
                          cka_fusion=cka_fusion)
            self.blocks.append(block)
        self.neck = nn.Sequential(
            nn.Conv2d(embed_dim, out_chans, kernel_size=1, bias=False),
            LayerNorm2d(out_chans),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(out_chans),
        )

    def forward(self, x: torch.Tensor, ema: bool = False) -> torch.Tensor:
        if self.cnn_branch:
            # CNN branch: n (before, after) pairs plus the last extra output.
            cnn_pairs, cnn_final = self.cnn_encoder(x)
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        cba_start = len(self.blocks) - self.cba_last_n
        for i, blk in enumerate(self.blocks):
            if blk.use_cba:
                before, after = cnn_pairs[i - cba_start]
                x = blk(x, ema=ema,
                        cnn_before=before.permute(0, 2, 3, 1),
                        cnn_after=after.permute(0, 2, 3, 1))
            else:
                x = blk(x, ema=ema)
        if self.cnn_branch:
            # Plain final ViT/CNN fusion (no CKA, no attention gate).
            x = x + cnn_final.permute(0, 2, 3, 1)
        x = self.neck(x.permute(0, 3, 1, 2))
        return x

    def cka_loss(self, clear: bool = True) -> torch.Tensor:
        """Aggregate the normalized ``L_CKA`` from the last forward.

        With ``n = cba_last_n`` CBA-enabled blocks there are ``2n`` CBA
        modules, each producing a ``(B, C)`` C_weight.  The CBA part is
        normalized exactly as

            (1 / (B * 2n * ((2n) * C))) * sum(C_weight over all CBAs)

        i.e. the literal ``batch_size * total_CBA_in_model *
        total_channels_in_all_CBA`` denominator (the CBA count is applied
        twice).

        Returns a scalar; returns zero when the model has no CBA modules.
        The stored tensors are cleared afterwards so their graphs are not
        retained across iterations.
        """
        cb_as = [m for m in self.modules() if isinstance(m, CrossBranchAttention)]
        n_cba = len(cb_as)
        device = self.neck[0].weight.device
        if n_cba == 0:
            return torch.zeros((), device=device)

        terms = [m.last_cka for m in cb_as if m.last_cka is not None]
        if clear:
            for m in cb_as:
                m.last_cka = None
        if not terms:
            return torch.zeros((), device=device)

        batch = terms[0].shape[0]
        channels = terms[0].shape[1]
        return sum(t.sum() for t in terms) / (batch * n_cba * (n_cba * channels))


# ============================================================
# Prompt encoder (repo segment_anything/modeling/prompt_encoder.py)
# ============================================================

class PositionEmbeddingRandom(nn.Module):
    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer("positional_encoding_gaussian_matrix",
                             scale * torch.randn((2, num_pos_feats)))

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * math.pi * coords
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        h, w = size
        device = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        # Determinism exception: `cumsum` has no deterministic CUDA kernel per
        # PyTorch, but here it runs over a constant all-ones grid (a sequential
        # prefix sum), so the result is exact/stable. Scoped warn-only keeps
        # strict mode for every other op.
        with deterministic_warn_only():
            y_embed = grid.cumsum(dim=0) - 0.5
            x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w
        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)

    def forward_with_coords(self, coords_input: torch.Tensor,
                            image_size: Tuple[int, int]) -> torch.Tensor:
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords.to(torch.float))


class PromptEncoder(nn.Module):
    def __init__(self, embed_dim: int, image_embedding_size: Tuple[int, int],
                 input_image_size: Tuple[int, int], mask_in_chans: int,
                 activation: Type[nn.Module] = nn.GELU) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.input_image_size = input_image_size
        self.image_embedding_size = image_embedding_size
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)
        self.num_point_embeddings: int = 4
        point_embeddings = [nn.Embedding(1, embed_dim) for _ in range(self.num_point_embeddings)]
        self.point_embeddings = nn.ModuleList(point_embeddings)
        self.not_a_point_embed = nn.Embedding(1, embed_dim)
        self.mask_input_size = (4 * image_embedding_size[0], 4 * image_embedding_size[1])
        self.mask_downscaling = nn.Sequential(
            nn.Conv2d(1, mask_in_chans // 4, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans // 4),
            activation(),
            nn.Conv2d(mask_in_chans // 4, mask_in_chans, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans),
            activation(),
            nn.Conv2d(mask_in_chans, embed_dim, kernel_size=1),
        )
        self.no_mask_embed = nn.Embedding(1, embed_dim)

    def get_dense_pe(self) -> torch.Tensor:
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)

    def _embed_boxes(self, boxes: torch.Tensor) -> torch.Tensor:
        boxes = boxes + 0.5
        coords = boxes.reshape(-1, 2, 2)
        corner_embedding = self.pe_layer.forward_with_coords(coords, self.input_image_size)
        corner_embedding[:, 0, :] += self.point_embeddings[2].weight
        corner_embedding[:, 1, :] += self.point_embeddings[3].weight
        return corner_embedding

    def forward(self, points, boxes, masks) -> Tuple[torch.Tensor, torch.Tensor]:
        bs = boxes.shape[0]
        sparse_embeddings = torch.empty((bs, 0, self.embed_dim), device=self.point_embeddings[0].weight.device)
        if boxes is not None:
            box_embeddings = self._embed_boxes(boxes)
            sparse_embeddings = torch.cat([sparse_embeddings, box_embeddings], dim=1)
        if masks is not None:
            dense_embeddings = self.mask_downscaling(masks)
        else:
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                bs, -1, self.image_embedding_size[0], self.image_embedding_size[1])
        return sparse_embeddings, dense_embeddings


# ============================================================
# Two-way transformer (repo segment_anything/modeling/transformer.py)
# ============================================================

class TwoWayAttention(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, downsample_rate: int = 1) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0
        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    def _separate_heads(self, x: torch.Tensor, num_heads: int) -> torch.Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)

    def _recombine_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)
        out = attn @ v
        out = self._recombine_heads(out)
        return self.out_proj(out)


class TwoWayAttentionBlock(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, mlp_dim: int = 2048,
                 activation: Type[nn.Module] = nn.ReLU, attention_downsample_rate: int = 2,
                 skip_first_layer_pe: bool = False) -> None:
        super().__init__()
        self.self_attn = TwoWayAttention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.cross_attn_token_to_image = TwoWayAttention(embedding_dim, num_heads,
                                                         downsample_rate=attention_downsample_rate)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.mlp = MLPBlock(embedding_dim, mlp_dim, activation)
        self.norm3 = nn.LayerNorm(embedding_dim)
        self.norm4 = nn.LayerNorm(embedding_dim)
        self.cross_attn_image_to_token = TwoWayAttention(embedding_dim, num_heads,
                                                         downsample_rate=attention_downsample_rate)
        self.skip_first_layer_pe = skip_first_layer_pe

    def forward(self, queries: torch.Tensor, keys: torch.Tensor,
                query_pe: torch.Tensor, key_pe: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.skip_first_layer_pe:
            queries = self.self_attn(q=queries, k=queries, v=queries)
        else:
            q = queries + query_pe
            attn_out = self.self_attn(q=q, k=q, v=queries)
            queries = queries + attn_out
        queries = self.norm1(queries)

        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm2(queries)

        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)

        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries)
        keys = keys + attn_out
        keys = self.norm4(keys)

        return queries, keys


class TwoWayTransformer(nn.Module):
    def __init__(self, depth: int, embedding_dim: int, num_heads: int,
                 mlp_dim: int, activation: Type[nn.Module] = nn.ReLU,
                 attention_downsample_rate: int = 2) -> None:
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.layers = nn.ModuleList()
        for i in range(depth):
            self.layers.append(TwoWayAttentionBlock(
                embedding_dim=embedding_dim, num_heads=num_heads, mlp_dim=mlp_dim,
                activation=activation, attention_downsample_rate=attention_downsample_rate,
                skip_first_layer_pe=(i == 0)))
        self.final_attn_token_to_image = TwoWayAttention(embedding_dim, num_heads,
                                                         downsample_rate=attention_downsample_rate)
        self.norm_final_attn = nn.LayerNorm(embedding_dim)

    def forward(self, image_embedding: torch.Tensor, image_pe: torch.Tensor,
                point_embedding: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        bs, c, h, w = image_embedding.shape
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
        image_pe = image_pe.flatten(2).permute(0, 2, 1)
        queries = point_embedding
        keys = image_embedding
        for layer in self.layers:
            queries, keys = layer(queries=queries, keys=keys,
                                  query_pe=point_embedding, key_pe=image_pe)
        q = queries + point_embedding
        k = keys + image_pe
        attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm_final_attn(queries)
        return queries, keys


# ============================================================
# Mask decoder (repo segment_anything/modeling/mask_decoder.py,
# num_mask_tokens = num_classes, 2-stage upsampling)
# ============================================================

class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_layers: int) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class MaskDecoder(nn.Module):
    def __init__(self, *, transformer_dim: int, transformer: nn.Module,
                 num_multimask_outputs: int = 3,
                 activation: Type[nn.Module] = nn.GELU,
                 iou_head_depth: int = 3, iou_head_hidden_dim: int = 256) -> None:
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer
        self.num_multimask_outputs = num_multimask_outputs
        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )
        self.output_hypernetworks_mlps = nn.ModuleList([
            MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
            for i in range(self.num_mask_tokens)])
        self.iou_prediction_head = MLP(transformer_dim, iou_head_hidden_dim,
                                       self.num_mask_tokens, iou_head_depth)

    def forward(self, image_embeddings: torch.Tensor, image_pe: torch.Tensor,
                sparse_prompt_embeddings: torch.Tensor, dense_prompt_embeddings: torch.Tensor,
                multimask_output: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        masks, iou_pred = self.predict_masks(
            image_embeddings=image_embeddings, image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings)
        return masks, iou_pred

    def predict_masks(self, image_embeddings: torch.Tensor, image_pe: torch.Tensor,
                      sparse_prompt_embeddings: torch.Tensor,
                      dense_prompt_embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)
        # One box prompt per image, so the batch can be processed at once
        # (repeat_interleave(1) instead of tokens.shape[0], which is only
        # valid for a batch of one).
        src = torch.repeat_interleave(image_embeddings, 1, dim=0)
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1:(1 + self.num_mask_tokens), :]
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)
        hyper_in_list = [self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :])
                         for i in range(self.num_mask_tokens)]
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)
        iou_pred = self.iou_prediction_head(iou_token_out)
        return masks, iou_pred


# ============================================================
# DSR-SAM modules (repo sam_lora_image_encoder.py)
# ============================================================

class _LoRA_qkv(nn.Module):
    """qkv linear with LoRA increments on q and v plus optional EMA copies."""

    def __init__(self, qkv: nn.Module, linear_a_q: nn.Module,
                 linear_b_q: nn.Module, linear_a_v: nn.Module,
                 linear_b_v: nn.Module,
                 ema_linear_a_q: Optional[nn.Module] = None,
                 ema_linear_b_q: Optional[nn.Module] = None,
                 ema_linear_a_v: Optional[nn.Module] = None,
                 ema_linear_b_v: Optional[nn.Module] = None) -> None:
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = qkv.in_features
        if ema_linear_a_q is not None:
            self.ema_linear_a_q = ema_linear_a_q
            self.ema_linear_b_q = ema_linear_b_q
            self.ema_linear_a_v = ema_linear_a_v
            self.ema_linear_b_v = ema_linear_b_v

    def _use_ema(self, ema: bool) -> bool:
        """Whether to use the EMA LoRA branch.

        True only if the caller requests `ema` AND this module actually
        owns EMA copies (i.e. the parent was built with ema_mode=True).
        With ema_mode=False the EMA attributes are never created, so any
        `ema=True` request safely falls back to the student LoRA.
        """
        return (bool(ema) and getattr(self, 'ema_linear_a_q', None) is not None
                and getattr(self, 'ema_linear_b_q', None) is not None
                and getattr(self, 'ema_linear_a_v', None) is not None
                and getattr(self, 'ema_linear_b_v', None) is not None)

    def forward(self, x: torch.Tensor, ema: bool = False) -> torch.Tensor:
        qkv = self.qkv(x)
        if self._use_ema(ema):
            new_q = self.ema_linear_b_q(self.ema_linear_a_q(x))
            new_v = self.ema_linear_b_v(self.ema_linear_a_v(x))
        else:
            new_q = self.linear_b_q(self.linear_a_q(x))
            new_v = self.linear_b_v(self.linear_a_v(x))
        qkv[:, :, :, :self.dim] += new_q
        qkv[:, :, :, -self.dim:] += new_v
        return qkv


def ema_update(model, rate):
    """EMA update of the LoRA weights (repo ema_update, alpha = 0.999).

    No-op for blocks without EMA copies (ema_mode=False) or plain
    nn.Linear qkv modules.
    """
    encoder = model.sam.image_encoder
    for block in encoder.blocks.children():
        qkv = block.attn.qkv
        if getattr(qkv, 'ema_linear_a_q', None) is None:
            continue
        if getattr(qkv, 'linear_a_q', None) is None:
            continue
        avg_model_params = (list(qkv.ema_linear_a_q.parameters())
                            + list(qkv.ema_linear_b_q.parameters())
                            + list(qkv.ema_linear_a_v.parameters())
                            + list(qkv.ema_linear_b_v.parameters()))
        model_params = (list(qkv.linear_a_q.parameters())
                        + list(qkv.linear_b_q.parameters())
                        + list(qkv.linear_a_v.parameters())
                        + list(qkv.linear_b_v.parameters()))
        for moving_avg_param, param in zip(avg_model_params, model_params):
            moving_avg_param.data = (rate * moving_avg_param.data
                                     + (1 - rate) * param.data.detach())


class LoRA_Sam(nn.Module):
    """Applies LoRA adaptation with optional EMA copies to the SAM image
    encoder (repo LoRA_Sam)."""

    def __init__(self, sam_model, r: int, lora_layer=None, ema_mode=True) -> None:
        super(LoRA_Sam, self).__init__()
        self.ema_mode = ema_mode

        assert r > 0
        if lora_layer:
            self.lora_layer = lora_layer
        else:
            self.lora_layer = list(range(len(sam_model.image_encoder.blocks)))

        # Freeze the prompt encoder and the pre-trained image encoder.
        for param in sam_model.prompt_encoder.parameters():
            param.requires_grad = False
        for param in sam_model.image_encoder.parameters():
            param.requires_grad = False
        # DSR-SAM2: the whole CNN branch (ResNet18 layers 1-3 pretrained,
        # layer4 and the extra ResBlocks from scratch), the 2*n CBA blocks
        # and every AttentionGate are trained jointly with the LoRA
        # increments. Everything else in the ViT stays frozen.
        for name, param in sam_model.image_encoder.named_parameters():
            if (name.startswith('cnn_encoder.') or '.cba' in name
                    or 'attention_gate' in name):
                param.requires_grad = True

        # Here, we do the surgery.
        for t_layer_i, blk in enumerate(sam_model.image_encoder.blocks):
            if t_layer_i not in self.lora_layer:
                continue
            w_qkv_linear = blk.attn.qkv
            self.dim = w_qkv_linear.in_features

            w_a_linear_q = nn.Linear(self.dim, r, bias=False)
            w_b_linear_q = nn.Linear(r, self.dim, bias=False)
            w_a_linear_v = nn.Linear(self.dim, r, bias=False)
            w_b_linear_v = nn.Linear(r, self.dim, bias=False)
            self.reset_A_parameters(w_a_linear_q)
            self.reset_B_parameters(w_b_linear_q)
            self.reset_A_parameters(w_a_linear_v)
            self.reset_B_parameters(w_b_linear_v)
            if self.ema_mode:
                ema_w_a_linear_q = copy.deepcopy(w_a_linear_q)
                ema_w_b_linear_q = copy.deepcopy(w_b_linear_q)
                ema_w_a_linear_v = copy.deepcopy(w_a_linear_v)
                ema_w_b_linear_v = copy.deepcopy(w_b_linear_v)
                # EMA weights are only updated by the moving-average rule.
                for p in (list(ema_w_a_linear_q.parameters()) + list(ema_w_b_linear_q.parameters())
                          + list(ema_w_a_linear_v.parameters()) + list(ema_w_b_linear_v.parameters())):
                    p.requires_grad = False
            else:
                ema_w_a_linear_q = ema_w_b_linear_q = ema_w_a_linear_v = ema_w_b_linear_v = None

            blk.attn.qkv = _LoRA_qkv(
                w_qkv_linear,
                w_a_linear_q, w_b_linear_q, w_a_linear_v, w_b_linear_v,
                ema_w_a_linear_q, ema_w_b_linear_q,
                ema_w_a_linear_v, ema_w_b_linear_v,
            )
        self.sam = sam_model

    def reset_A_parameters(self, w_A) -> None:
        nn.init.orthogonal_(w_A.weight)

    def reset_B_parameters(self, w_B) -> None:
        nn.init.zeros_(w_B.weight)

    def resolve_ema(self, ema: bool) -> bool:
        """Resolve a requested `ema` branch against this model's mode.

        Returns True only if the caller asks for EMA AND the model was
        built with ema_mode=True. With ema_mode=False every caller
        (train/val/test) is forced onto the student LoRA.
        """
        return bool(ema) and bool(getattr(self, 'ema_mode', False))

    def forward(self, batched_input, multimask_output, image_size,
                bbox_input=None, ema=False):
        # ema_mode=False -> always student LoRA, even if caller passes ema=True.
        ema = self.resolve_ema(ema)
        return self.sam(batched_input, multimask_output, image_size,
                        bbox_input=bbox_input, ema=ema)


# ============================================================
# SAM model (repo segment_anything/modeling/sam.py, forward_train
# extended with bounding-box prompts from the benchmark convention)
# ============================================================

class Sam(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(self, image_encoder, prompt_encoder, mask_decoder,
                 pixel_mean=[0.0, 0.0, 0.0], pixel_std=[1.0, 1.0, 1.0]) -> None:
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

    @property
    def device(self) -> Any:
        return self.pixel_mean.device

    def forward(self, batched_input, multimask_output, image_size,
                bbox_input=None, ema=False):
        return self.forward_train(batched_input, multimask_output, image_size,
                                  bbox_input=bbox_input, ema=ema)

    def forward_train(self, batched_input, multimask_output, image_size,
                      bbox_input=None, ema=False):
        input_images = self.preprocess(batched_input)
        image_embeddings = self.image_encoder(input_images, ema=ema)
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=None, boxes=bbox_input, masks=None)
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output)
        masks = self.postprocess_masks(
            low_res_masks, input_size=(image_size, image_size),
            original_size=(image_size, image_size))
        outputs = {
            'masks': masks,
            'iou_predictions': iou_predictions,
            'low_res_logits': low_res_masks,
            # Scalar L_CKA from this (student) forward; see
            # ImageEncoderViT.cka_loss.  Always present, zero without CBAs.
            'cka_loss': self.image_encoder.cka_loss(),
        }
        return outputs

    def postprocess_masks(self, masks, input_size, original_size) -> torch.Tensor:
        masks = F.interpolate(masks, (self.image_encoder.img_size, self.image_encoder.img_size),
                              mode="bilinear", align_corners=False)
        masks = masks[..., :input_size[0], :input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.pixel_mean) / self.pixel_std
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x


# ============================================================
# Model builder with SAM checkpoint loading (repo build_sam.py load_from)
# ============================================================

VIT_CONFIGS = {
    'vit_b': dict(
        encoder_embed_dim=768, encoder_depth=12, encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
    ),
    'vit_h': dict(
        encoder_embed_dim=1280, encoder_depth=32, encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
    ),
}


def _load_from(sam, state_dict, image_size, vit_patch_size, encoder_global_attn_indexes):
    ega = encoder_global_attn_indexes
    sam_dict = sam.state_dict()
    except_keys = ['mask_tokens', 'output_hypernetworks_mlps', 'iou_prediction_head']
    new_state_dict = {k: v for k, v in state_dict.items()
                      if k in sam_dict.keys()
                      and except_keys[0] not in k
                      and except_keys[1] not in k
                      and except_keys[2] not in k}
    pos_embed = new_state_dict['image_encoder.pos_embed']
    token_size = int(image_size // vit_patch_size)
    if pos_embed.shape[1] != token_size:
        pos_embed = pos_embed.permute(0, 3, 1, 2)
        pos_embed = F.interpolate(pos_embed, (token_size, token_size),
                                  mode='bilinear', align_corners=False)
        pos_embed = pos_embed.permute(0, 2, 3, 1)
        new_state_dict['image_encoder.pos_embed'] = pos_embed
        rel_pos_keys = [k for k in sam_dict.keys() if 'rel_pos' in k]
        global_rel_pos_keys = []
        for rel_pos_key in rel_pos_keys:
            num = int(rel_pos_key.split('.')[2])
            if num in ega:
                global_rel_pos_keys.append(rel_pos_key)
        for k in global_rel_pos_keys:
            rel_pos_params = new_state_dict[k]
            h, w = rel_pos_params.shape
            rel_pos_params = rel_pos_params.unsqueeze(0).unsqueeze(0)
            rel_pos_params = F.interpolate(rel_pos_params, (token_size * 2 - 1, w),
                                           mode='bilinear', align_corners=False)
            new_state_dict[k] = rel_pos_params[0, 0, ...]
    sam_dict.update(new_state_dict)
    sam.load_state_dict(sam_dict)


def _build_sam(image_size, num_classes, checkpoint=None, model_type='vit_h',
               cnn_pretrained=True, cba_last_n=4,
               cnn_branch=True, cka_fusion=True,
               use_attention_gate=True,
               use_attention_gate_cba=True,
               attention_gate_channels=None):
    prompt_embed_dim = 256
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    cfg = dict(VIT_CONFIGS[model_type])
    sam = Sam(
        image_encoder=ImageEncoderViT(
            depth=cfg['encoder_depth'], embed_dim=cfg['encoder_embed_dim'],
            img_size=image_size, mlp_ratio=4,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            num_heads=cfg['encoder_num_heads'], patch_size=vit_patch_size,
            qkv_bias=True, use_rel_pos=True,
            global_attn_indexes=cfg['encoder_global_attn_indexes'],
            window_size=14, out_chans=prompt_embed_dim,
            cnn_pretrained=cnn_pretrained, cba_last_n=cba_last_n,
            cnn_branch=cnn_branch, cka_fusion=cka_fusion,
            use_attention_gate=use_attention_gate,
            use_attention_gate_cba=use_attention_gate_cba,
            attention_gate_channels=attention_gate_channels,
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=num_classes,
            transformer=TwoWayTransformer(
                depth=2, embedding_dim=prompt_embed_dim,
                mlp_dim=2048, num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3, iou_head_hidden_dim=256,
        ),
        pixel_mean=[0.0, 0.0, 0.0],
        pixel_std=[1.0, 1.0, 1.0],
    )
    sam.train()
    if checkpoint is not None:
        with open(checkpoint, 'rb') as f:
            state_dict = torch.load(f, map_location='cpu')
        _load_from(sam, state_dict, image_size, vit_patch_size,
                   cfg['encoder_global_attn_indexes'])
    return sam, image_embedding_size


def build_dsr_sam2(checkpoint=None, model_type='vit_b', image_size=256,
                 num_classes=1, rank=64, ema_mode=True,
                 cnn_pretrained=True, cba_last_n=4,
                 cnn_branch=True, cka_fusion=True,
                 use_attention_gate=True,
                 use_attention_gate_cba=True,
                 attention_gate_channels=None):
    """Build DSR-SAM2: SAM ViT + ResNet18 CNN branch + 2*n CBA blocks, wrapped
    by the DSR-SAM LoRA/EMA wrapper.

    ``cnn_branch``: True builds the dual-branch model. False removes the
    ResNet18 branch, the CBA blocks and the final CNN fusion, i.e. plain
    SAM + LoRA (ablation control).

    ``cka_fusion``: CBA combination rule. ``True`` (default) uses
    ``f_vit + ((1 - C_weight) * f_cnn + f_gated_cnn) / 2``; ``False`` is the
    unweighted ``f_vit + f_cnn`` and bypasses the gate.

    ``use_attention_gate``: master switch for the CBA AttentionGate
    (Oktay et al. 2018); ``use_attention_gate_cba`` selects the 2n CBA gates.
    ``attention_gate_channels`` is the gate's intermediate width ``F_int``
    (None -> C, 768 for vit_b).  The final ViT/CNN fusion is plain
    ``x = x + cnn_final`` (no CKA, no gate).

    ``cba_last_n = n`` adds ``n`` extra CNN ResBlocks and enables CBA in the
    last ``n`` ViT blocks (one additional ResBlock per CBA-enabled block).

    Every forward also exposes the scalar ``L_CKA`` in ``outputs['cka_loss']``:
    the sum of all ``2n`` CBA ``C_weight`` terms normalized by
    ``batch * (2n) * ((2n) * C)``.  Training adds
    ``cka_loss_weight * L_CKA`` to the segmentation loss.
    """
    assert model_type in VIT_CONFIGS
    sam, _ = _build_sam(image_size=image_size, num_classes=num_classes,
                        checkpoint=checkpoint, model_type=model_type,
                        cnn_pretrained=cnn_pretrained, cba_last_n=cba_last_n,
                        cnn_branch=cnn_branch,
                        cka_fusion=cka_fusion,
                        use_attention_gate=use_attention_gate,
                        use_attention_gate_cba=use_attention_gate_cba,
                        attention_gate_channels=attention_gate_channels)
    net = LoRA_Sam(sam, r=rank, ema_mode=ema_mode)
    return net
