"""DSR-SAM2: dual-branch SAM with Grassmann cross-branch attention.

Architecture (input 256x256 -> ViT 16x16 tokens):

  ViT branch (unchanged SAM ViT-B):
      patch embed 16x16 -> 12 transformer blocks (window/global attention)
      -> BUSSAM-style fusion x + 0.5 * f_cnn -> SAM neck (256ch).

  CNN branch (ResNet18, IMAGENET1K_V1):
      stem/layer1/layer2/layer3 -> 16x16x256
      -> layer4 (new 256->768 ResBlock, 16x16x768)
      -> layer5..layer8 (new 768->768 ResBlocks, 16x16x768).

  Cross-branch attention (8 CBA blocks):
      ViT blocks 8,9,10,11 pair with CNN layers 5,6,7,8.
      Each ViT block has two CBA blocks:
        * after norm1, with the resblock "before" feature;
        * after norm2, with the resblock "after" feature.
      Each CBA computes a per-channel Grassmann geodesic score G[c]
      (mean over ViT channels of arccos(|cos|), in [0, pi/2]) and fuses
          (1 - G[c]) * f_vit[..., c] + G[c] * f_cnn[..., c]
      with G[c] normalized to [0, 1] by dividing the geodesic distance by pi/2.
      Optionally, the s smallest singular values/vectors of both branch
      features are removed before G is computed
      (truncate_before_compute_grassmann=s); truncate_ref=True uses those
      truncated copies only as the G reference, while truncate_ref=False
      fuses them too.
      before running BUSSAM's spatial-attention + MLP adapter.

  Everything else (LoRA/EMA/subspace truncation, prompt encoder, mask
  decoder, L_tsd) is inherited from DSR-SAM.
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
      * ``cba2`` acts on ``norm2(x)`` after the MLP (residual add, scale 0.5).

    The ViT/CNN fusion itself is not BUSSAM's direct ``f_vit + f_cnn``: it is
    the per-channel Grassmann-weighted blend described in
    ``CrossBranchAttention``.
    """

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 qkv_bias: bool = True, norm_layer: Type[nn.Module] = nn.LayerNorm,
                 act_layer: Type[nn.Module] = nn.GELU, use_rel_pos: bool = False,
                 rel_pos_zero_init: bool = True, window_size: int = 0,
                 input_size: Optional[Tuple[int, int]] = None,
                 use_cba: bool = False, detach_g: bool = True,
                 truncate_before_compute_grassmann: int = 0,
                 truncate_ref: bool = False) -> None:
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
                detach_g=detach_g,
                truncate_before_compute_grassmann=truncate_before_compute_grassmann,
                truncate_ref=truncate_ref)
            self.cba2 = CrossBranchAttention(
                dim, mlp_ratio=0.25, act_layer=act_layer, skip_connect=False,
                detach_g=detach_g,
                truncate_before_compute_grassmann=truncate_before_compute_grassmann,
                truncate_ref=truncate_ref)
            # BUSSAM: x = x + scale * adapter(norm2(x), cnnx), scale = 0.5
            self.cba2_scale = 0.5

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
            # CBA n.2 of this block + BUSSAM-style 0.5 residual, after the MLP.
            x = x + self.cba2_scale * self.cba2(xn, cnn_after)
        return x


# ============================================================
# DSR-SAM2 CNN branch, Grassmann score and CBA (BUSSAM cross-branch
# attention fused by the per-channel Grassmann geodesic distance)
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
    one 256->768 ResBlock (layer4) and four 768->768 ResBlocks (layer5-8),
    all at 16x16 for a 256x256 input.

    layer4-8 are new (no pretrained weights). layer5-8 are consumed by the
    last four ViT blocks: their *before* feature feeds CBA n.1 and their
    *after* feature feeds CBA n.2, so 4 blocks x 2 = 8 CBA blocks.
    """

    def __init__(self, pretrained: bool = True, out_chans: int = 768) -> None:
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
        # New branch-specific layers (no pretrained weights).
        self.layer4 = BasicResBlock(256, out_chans, stride=1)   # 16x16x768
        self.layer5 = BasicResBlock(out_chans, out_chans)       # 16x16x768
        self.layer6 = BasicResBlock(out_chans, out_chans)
        self.layer7 = BasicResBlock(out_chans, out_chans)
        self.layer8 = BasicResBlock(out_chans, out_chans)
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
        for layer in (self.layer5, self.layer6, self.layer7, self.layer8):
            before = x
            x = layer(x)
            pairs.append((before, x))
        return pairs, x


def per_channel_grassmann_score(f_vit: torch.Tensor, f_cnn: torch.Tensor,
                                eps: float = 1e-6,
                                detach: bool = True) -> torch.Tensor:
    """Per-CNN-channel mean Grassmann geodesic distance.

    ``f_vit`` and ``f_cnn`` are ``(B, H, W, C)`` (here ``B, 16, 16, 768``).
    Each channel is the ``H*W``-dimensional spatial vector, i.e. a 1-D
    subspace.  With ``cos_ij = <v_i, n_j> / (||v_i|| ||n_j||)``:

        G_score[j] = mean_i arccos(|cos_ij|)        in [0, pi/2] rad

    i.e. every CNN channel ``j`` is compared with all C ViT channels and the
    C geodesic distances are averaged.  The whole C x C cosine matrix is one
    batched matmul (no Python loop), which is the speed-optimized form.

    Returns ``(B, C)``.  ``detach=True`` blocks gradients through the metric
    (the score is used as a fixed gate); ``detach=False`` lets them flow.
    """
    if f_vit.shape[1:3] != f_cnn.shape[1:3]:
        # Defensive: the CNN and ViT maps must have the same spatial grid.
        f_cnn = F.interpolate(f_cnn.permute(0, 3, 1, 2),
                              size=f_vit.shape[1:3], mode='bilinear',
                              align_corners=False).permute(0, 2, 3, 1)
    if detach:
        f_vit = f_vit.detach()
        f_cnn = f_cnn.detach()
    # arccos is evaluated in fp32 even if the branches run under autocast.
    f_vit = f_vit.float()
    f_cnn = f_cnn.float()
    b, h, w, c = f_vit.shape
    v = f_vit.reshape(b, h * w, c)
    n = f_cnn.reshape(b, h * w, c)
    v = v / v.norm(dim=1, keepdim=True).clamp_min(eps)
    n = n / n.norm(dim=1, keepdim=True).clamp_min(eps)
    cos = torch.bmm(v.transpose(1, 2), n).abs()          # (B, C_vit, C_cnn)
    if detach:
        cos = cos.clamp(0.0, 1.0)
    else:
        # arccos'(1) is infinite; stay a hair away when gradients flow.
        cos = cos.clamp(0.0, 1.0 - 1e-6)
    g = torch.arccos(cos).mean(dim=1)                    # mean over ViT channels
    return g


def truncate_lowest_sigma(x: torch.Tensor, n_remove: int) -> torch.Tensor:
    """Truncate the ``n_remove`` smallest singular values of each sample's
    ``(H*W, C)`` feature matrix and reconstruct.

    ``x`` is ``(B, H, W, C)``.  Per sample, ``X = U diag(S) V^T``; the lowest
    ``n_remove`` entries of ``S`` and their matching ``U`` columns / ``V^T``
    rows are zeroed, then ``X_trunc = (U_z * S_z) @ Vh_z``.  This is the same
    operation used by the per-channel Grassmann analysis notebook.
    """
    b, h, w, c = x.shape
    x2 = x.reshape(b, h * w, c).float()
    u, s, vh = torch.linalg.svd(x2, full_matrices=False)
    keep = max(s.shape[-1] - int(n_remove), 0)
    s_z = s.clone()
    s_z[..., keep:] = 0.0
    u_z = u.clone()
    u_z[..., :, keep:] = 0.0
    vh_z = vh.clone()
    vh_z[..., keep:, :] = 0.0
    x_trunc = (u_z * s_z.unsqueeze(-2)) @ vh_z
    return x_trunc.reshape(b, h, w, c).to(x.dtype)


class CrossBranchAttention(nn.Module):
    """CBA block: Grassmann-weighted ViT/CNN fusion + BUSSAM MLP adapter.

    BUSSAM's MLPAdapter starts with ``x = f_vit + f_cnn``.  DSR-SAM2 replaces
    that sum by, for every channel c (with G' = G_score / (pi/2) in [0, 1]),

        fused[..., c] = (1 - G'[c]) * f_vit[..., c]
                        +     G'[c]  * f_cnn[..., c]

    and then continues with BUSSAM's SpatialAttention + Linear/GELU/Linear.
    With ``skip_connect=False`` the returned tensor is the adapter output
    (the owning ``Block`` applies the residual, exactly like BUSSAM).

    ``truncate_before_compute_grassmann = s > 0`` truncates the s smallest
    singular values before the Grassmann score.  With ``truncate_ref=True``
    the truncated copies are only the reference used to compute G_score and
    the fused features stay original; with ``truncate_ref=False`` the
    truncated features are also the ones fused.
    """

    def __init__(self, dim: int, mlp_ratio: float = 0.25,
                 act_layer: Type[nn.Module] = nn.GELU,
                 skip_connect: bool = False, detach_g: bool = True,
                 truncate_before_compute_grassmann: int = 0,
                 truncate_ref: bool = False) -> None:
        super().__init__()
        self.skip_connect = skip_connect
        self.detach_g = detach_g
        self.truncate_before_compute_grassmann = max(
            0, int(truncate_before_compute_grassmann))
        # truncate_ref only has an effect when truncation is enabled.
        self.truncate_ref = bool(truncate_ref)
        hidden = int(dim * mlp_ratio)
        self.act = act_layer()
        self.D_fc1 = nn.Linear(dim, hidden)
        self.D_fc2 = nn.Linear(hidden, dim)
        self.spatial_attention = SpatialAttention()

    def forward(self, f_vit: torch.Tensor, f_cnn: torch.Tensor,
                detach_g: Optional[bool] = None) -> torch.Tensor:
        detach_g = self.detach_g if detach_g is None else detach_g
        n_trunc = self.truncate_before_compute_grassmann
        if n_trunc > 0:
            # truncate_ref=True only needs a reference for G_score, so when
            # G is detached we can drop those SVD graphs immediately.
            src_vit, src_cnn = f_vit, f_cnn
            if self.truncate_ref and detach_g:
                src_vit, src_cnn = f_vit.detach(), f_cnn.detach()
            f_vit_ref = truncate_lowest_sigma(src_vit, n_trunc)
            f_cnn_ref = truncate_lowest_sigma(src_cnn, n_trunc)
        else:
            f_vit_ref, f_cnn_ref = f_vit, f_cnn

        g = per_channel_grassmann_score(f_vit_ref, f_cnn_ref, detach=detach_g)
        g = g[:, None, None, :]                       # (B, 1, 1, C)

        # truncate_ref=False: the truncated features are fused as well.
        if n_trunc > 0 and not self.truncate_ref:
            f_vit, f_cnn = f_vit_ref, f_cnn_ref

        fused = (1.0 - g * (2.0 / math.pi)) * f_vit + g * (2.0 / math.pi) * f_cnn
        xs = self.spatial_attention(fused.permute(0, 3, 1, 2))
        xs = xs.permute(0, 2, 3, 1)
        xs = self.D_fc2(self.act(self.D_fc1(xs)))
        if self.skip_connect:
            fused = fused + xs
        else:
            fused = xs
        return fused


class ImageEncoderViT(nn.Module):
    """SAM ViT encoder (unchanged) + ResNet18 CNN branch + 8 CBA blocks.

    Input goes through both branches in parallel:

      * ViT branch: the original SAM encoder (patch embed, 12 blocks, neck).
      * CNN branch: ResNet18 ImageNet1k-v1 through layer3 (16x16x256), then
        ResBlock layer4 (16x16x768) and ResBlocks layer5-8 (16x16x768).
      * Last ``cba_last_n`` ViT blocks (8, 9, 10, 11 for vit_b) each own two
        CBA blocks.  Block 8 pairs with layer5, 9 with layer6, 10 with layer7,
        11 with layer8: layer before -> CBA after norm1, layer after -> CBA
        after norm2.  Total = 8 CBA blocks.
      * Before the SAM decoder the branches are added as in BUSSAM:
        ``x = x + 0.5 * cnn_final`` followed by the frozen SAM neck.
    """

    def __init__(self, img_size: int = 1024, patch_size: int = 16, in_chans: int = 3,
                 embed_dim: int = 768, depth: int = 12, num_heads: int = 12,
                 mlp_ratio: float = 4.0, out_chans: int = 256, qkv_bias: bool = True,
                 norm_layer: Type[nn.Module] = nn.LayerNorm, act_layer: Type[nn.Module] = nn.GELU,
                 use_abs_pos: bool = True, use_rel_pos: bool = False,
                 rel_pos_zero_init: bool = True, window_size: int = 0,
                 global_attn_indexes: Tuple[int, ...] = (),
                 cnn_pretrained: bool = True, cba_last_n: int = 4,
                 grassmann_detach: bool = True,
                 truncate_before_compute_grassmann: int = 0,
                 truncate_ref: bool = False) -> None:
        super().__init__()
        self.img_size = img_size
        self.cnn_encoder = ResNet18Branch(pretrained=cnn_pretrained,
                                          out_chans=embed_dim)
        self.patch_embed = PatchEmbed(kernel_size=(patch_size, patch_size),
                                      stride=(patch_size, patch_size),
                                      in_chans=in_chans, embed_dim=embed_dim)
        self.pos_embed = None
        if use_abs_pos:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, img_size // patch_size, img_size // patch_size, embed_dim)
            )
        if cba_last_n > 4:
            raise ValueError('cba_last_n must be <= 4: only CNN layers 5-8 '
                             'provide before/after CBA features')
        self.cba_last_n = max(0, min(cba_last_n, depth))
        self.grassmann_detach = grassmann_detach
        self.blocks = nn.ModuleList()
        for i in range(depth):
            use_cba = i >= depth - self.cba_last_n
            block = Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                          qkv_bias=qkv_bias, norm_layer=norm_layer, act_layer=act_layer,
                          use_rel_pos=use_rel_pos, rel_pos_zero_init=rel_pos_zero_init,
                          window_size=window_size if i not in global_attn_indexes else 0,
                          input_size=(img_size // patch_size, img_size // patch_size),
                          use_cba=use_cba, detach_g=grassmann_detach,
                          truncate_before_compute_grassmann=truncate_before_compute_grassmann,
                          truncate_ref=truncate_ref)
            self.blocks.append(block)
        self.neck = nn.Sequential(
            nn.Conv2d(embed_dim, out_chans, kernel_size=1, bias=False),
            LayerNorm2d(out_chans),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(out_chans),
        )

    def forward(self, x: torch.Tensor, ema: bool = False) -> torch.Tensor:
        # CNN branch: (before, after) for layers 5-8 plus the layer8 output.
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
        # BUSSAM final fusion: x = x + 0.5 * cnnx (then the frozen SAM neck).
        x = x + 0.5 * cnn_final.permute(0, 2, 3, 1)
        x = self.neck(x.permute(0, 3, 1, 2))
        return x


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

class _LoRAtruncation_qkv(nn.Module):
    """qkv linear with LoRA increments on q and v plus EMA copies and
    SVD-based subspace truncation of the base weight (repo)."""

    def __init__(self, qkv: nn.Module, linear_a_q: nn.Module, linear_b_q: nn.Module,
                 linear_a_v: nn.Module, linear_b_v: nn.Module,
                 ema_linear_a_q: Optional[nn.Module] = None,
                 ema_linear_b_q: Optional[nn.Module] = None,
                 ema_linear_a_v: Optional[nn.Module] = None,
                 ema_linear_b_v: Optional[nn.Module] = None,
                 index: int = 8,
                 top_s_change_rates: Optional[int] = None,
                 d_delta: bool = False,
                 alpha: float = 0.1) -> None:
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = qkv.in_features
        self.eps = 1e-8
        if ema_linear_a_q is not None:
            self.ema_linear_a_q = ema_linear_a_q
            self.ema_linear_b_q = ema_linear_b_q
            self.ema_linear_a_v = ema_linear_a_v
            self.ema_linear_b_v = ema_linear_b_v
        self.index = index
        # Number of top change rates that contribute to L_tsd. Set by the
        # `top_s_change_rates` config; defaults to `index` (truncation_size)
        # when not provided.
        self.top_s_change_rates = index if top_s_change_rates is None else top_s_change_rates
        # `d_delta` (config): when True, L_tsd's numerator keeps only the
        # change rates that increased since the previous iteration
        # (change_rate_t - change_rate_{t-1} > 0). The previous iteration's
        # LoRA (A, B) weights are cached in `_prev_lora_{q,v}`, re-projected
        # onto the current base-weight SVD on the next step, then released.
        # When False, L_tsd stays the unconditional mean over the top-s rates.
        self.d_delta = d_delta
        # `alpha` (config): coefficient of the logarithmic L_tsd loss
        # `log(1 + alpha * change_rate)` used by `compute_l_tsd` in place of
        # the divide-by-max normalization.
        self.alpha = alpha
        self._prev_lora_q = None
        self._prev_lora_v = None
        self._cached_svd_q = None
        self._cached_svd_v = None
        self._svd_cache_valid = False

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

    def calculate_change_rate(self, a, bb, r):
        # Paper Eq. 2: delta_i = u_i^T delta_W v_i / (sigma_i + eps).
        change_rate = torch.abs(bb) / (torch.abs(a) + self.eps)
        # Check that change_rate is 1D
        if change_rate.dim() != 1:
            raise ValueError(f"Expected change_rate to be 1D, got shape {change_rate.shape}")
        _, top_r_indices = torch.topk(change_rate, r)
        return change_rate, top_r_indices

    def calculate_l_tsd_rate(self, a, bb, r):
        # L_tsd rate that enters the loss (no absolute value; `a` is the
        # base-weight singular value sigma, so the denominator carries no
        # deltaW term):
        #     change_rate = ( bb / (a + eps) ) ** 2
        change_rate = (bb / (a + self.eps)) ** 2
        # Check that change_rate is 1D
        if change_rate.dim() != 1:
            raise ValueError(f"Expected change_rate to be 1D, got shape {change_rate.shape}")
        _, top_r_indices = torch.topk(change_rate, r)
        return change_rate, top_r_indices

    def _project_change_rate(self, a_weight, b_weight, weight_sigma, weight_u, weight_vt):
        """Project a LoRA delta (delta_W = B @ A) onto a cached base SVD.

        Returns the 1D per-singular-direction change rates, matching the
        current-iteration rates in ``compute_l_tsd`` (same base-weight basis).
        """
        delta_W = b_weight @ a_weight
        delta_sigma = torch.diag(torch.matmul(torch.matmul(weight_u.T, delta_W), weight_vt.T))
        return torch.abs(delta_sigma) / (torch.abs(weight_sigma) + self.eps)

    def _select_change_rates(self, change_rate, top_idx, prev_lora,
                             weight_sigma, weight_u, weight_vt):
        """Indices (into ``change_rate``) that contribute to L_tsd.

        Without ``d_delta`` -- or before the first cached iteration -- this is
        the full top-s index set (plain unconditional L_tsd numerator). With
        ``d_delta`` and a cached previous LoRA, only the top-s directions whose
        change rate increased (change_rate_t - change_rate_{t-1} > 0) are kept.
        """
        if not self.d_delta or prev_lora is None:
            return top_idx
        with torch.no_grad():
            prev_rate = self._project_change_rate(
                prev_lora[0], prev_lora[1], weight_sigma, weight_u, weight_vt)
            keep = (change_rate.detach() - prev_rate) > 0
        return top_idx[keep[top_idx]]

    def cache_base_weight_svd(self, force=False):
        """Cache SVD of base weights (q and v projections) for L_tsd computation.

        The SVD depends only on the frozen base weight, which is modified sole-
        ly by ``update_base_weight`` (which invalidates this cache). A valid
        cache therefore always matches the current base weight, so recomputing
        would return identical tensors and can be skipped. ``force=True``
        recomputes unconditionally.
        """
        if self._svd_cache_valid and not force:
            return
        device = self.qkv.weight.device
        base_W_q = self.qkv.weight[:self.dim, :]
        weight_u_q, weight_sigma_q, weight_vt_q = torch.linalg.svd(base_W_q, full_matrices=False)
        self._cached_svd_q = (weight_u_q, weight_sigma_q, weight_vt_q)
        base_W_v = self.qkv.weight[-self.dim:, :]
        weight_u_v, weight_sigma_v, weight_vt_v = torch.linalg.svd(base_W_v, full_matrices=False)
        self._cached_svd_v = (weight_u_v, weight_sigma_v, weight_vt_v)
        self._svd_cache_valid = True

    def compute_l_tsd(self, ema=False):
        """Compute L_tsd: the log-squashed rate without deltaW in the denominator.

        Per branch (q, v):

            * the top-s directions are selected with the ORIGINAL change rate,
              ``|bb| / (|a| + eps)`` (including the d_delta growth filter);
            * the value that enters the loss is the rate
              ``(bb / (a + eps)) ** 2`` (no absolute values), where ``a`` is the
              base-weight singular value sigma (``deltaW`` is NOT in the
              denominator; the base SVD terms stay frozen).

        The kept rates are squashed with the logarithm:
            L_tsd = mean(log(1 + alpha * change_rate))
        where ``alpha`` is the config coefficient (`self.alpha`, default 0.1).

        With `d_delta=True`, only top-s directions that grew since the previous
        iteration are kept. With `d_delta=False` the full top-s set is used.
        """
        device = self.qkv.weight.device
        total_l_tsd = torch.tensor(0.0, device=device)
        count = 0
        use_ema = self._use_ema(ema)

        # Compute for q
        if use_ema:
            a_q = self.ema_linear_a_q.weight
            b_q = self.ema_linear_b_q.weight
        else:
            a_q = self.linear_a_q.weight
            b_q = self.linear_b_q.weight
        delta_W_q = b_q @ a_q

        if not self._svd_cache_valid:
            self.cache_base_weight_svd()
        weight_u_q, weight_sigma_q, weight_vt_q = self._cached_svd_q
        delta_sigma_q = torch.diag(torch.matmul(torch.matmul(weight_u_q.T, delta_W_q), weight_vt_q.T))
        # Top-s selection still uses the ORIGINAL change rate.
        change_rate_q, top_q = self.calculate_change_rate(
            weight_sigma_q, delta_sigma_q, self.top_s_change_rates)
        # L_tsd rate (without deltaW) supplies the value that enters the loss.
        l_tsd_rate_q, _ = self.calculate_l_tsd_rate(
            weight_sigma_q, delta_sigma_q, self.top_s_change_rates)
        selected_q = self._select_change_rates(
            change_rate_q, top_q, self._prev_lora_q,
            weight_sigma_q, weight_u_q, weight_vt_q)
        if selected_q.numel() > 0:
            # No divide-by-max: squash the kept rates with
            # log(1 + alpha * rate).
            l_tsd_q = torch.log1p(self.alpha * l_tsd_rate_q[selected_q]).mean()
        else:
            l_tsd_q = torch.tensor(0.0, device=device)
        if self.d_delta:
            self._prev_lora_q = (a_q.detach().clone(), b_q.detach().clone())
        if not torch.isfinite(l_tsd_q):
            l_tsd_q = torch.tensor(0.0, device=device)
        total_l_tsd = total_l_tsd + l_tsd_q
        count += 1

        # Compute for v
        if use_ema:
            a_v = self.ema_linear_a_v.weight
            b_v = self.ema_linear_b_v.weight
        else:
            a_v = self.linear_a_v.weight
            b_v = self.linear_b_v.weight
        delta_W_v = b_v @ a_v

        if not self._svd_cache_valid:
            self.cache_base_weight_svd()
        weight_u_v, weight_sigma_v, weight_vt_v = self._cached_svd_v
        delta_sigma_v = torch.diag(torch.matmul(torch.matmul(weight_u_v.T, delta_W_v), weight_vt_v.T))
        # Top-s selection still uses the ORIGINAL change rate.
        change_rate_v, top_v = self.calculate_change_rate(
            weight_sigma_v, delta_sigma_v, self.top_s_change_rates)
        # L_tsd rate (without deltaW) supplies the value that enters the loss.
        l_tsd_rate_v, _ = self.calculate_l_tsd_rate(
            weight_sigma_v, delta_sigma_v, self.top_s_change_rates)
        selected_v = self._select_change_rates(
            change_rate_v, top_v, self._prev_lora_v,
            weight_sigma_v, weight_u_v, weight_vt_v)
        if selected_v.numel() > 0:
            # No divide-by-max: squash the kept rates with
            # log(1 + alpha * rate).
            l_tsd_v = torch.log1p(self.alpha * l_tsd_rate_v[selected_v]).mean()
        else:
            l_tsd_v = torch.tensor(0.0, device=device)
        if self.d_delta:
            self._prev_lora_v = (a_v.detach().clone(), b_v.detach().clone())
        if not torch.isfinite(l_tsd_v):
            l_tsd_v = torch.tensor(0.0, device=device)
        total_l_tsd = total_l_tsd + l_tsd_v
        count += 1

        result = total_l_tsd / count if count > 0 else torch.tensor(0.0, device=device)
        if not torch.isfinite(result):
            result = torch.tensor(0.0, device=device)
        return result

    def update_base_weight(self, ema=False):
        """Identify the top-`index` TSDs and truncate them from W (paper Eq. 3).

        ema=True uses the EMA LoRA delta; with ema_mode=False (no EMA
        copies) this falls back to the student LoRA delta.
        """
        device = self.qkv.weight.device
        use_ema = self._use_ema(ema)
        # Reuse the cached SVD of the base weight when still valid. The base
        # weight is modified only here (which invalidates the cache below), so
        # a valid cache is bit-identical to recomputing torch.linalg.svd.
        if not self._svd_cache_valid:
            self.cache_base_weight_svd()
        if use_ema:
            delta_W_q = self.ema_linear_b_q.weight @ self.ema_linear_a_q.weight
        else:
            delta_W_q = self.linear_b_q.weight @ self.linear_a_q.weight
        weight_u_q, weight_sigma_q, weight_vt_q = self._cached_svd_q
        delta_sigma_q = torch.diag(torch.matmul(torch.matmul(weight_u_q.T, delta_W_q), weight_vt_q.T))
        _, top_index_q = self.calculate_change_rate(weight_sigma_q, delta_sigma_q, self.index)
        remain_index_q = torch.tensor([idx for idx in range(weight_u_q.shape[1])
                                       if idx not in top_index_q], device=device)
        new_base_W_q = (weight_u_q[:, remain_index_q] @ torch.diag(weight_sigma_q[remain_index_q])
                        @ weight_vt_q[remain_index_q, :])
        self.qkv.weight[:self.dim, :] = new_base_W_q.clone()

        if use_ema:
            delta_W_v = self.ema_linear_b_v.weight @ self.ema_linear_a_v.weight
        else:
            delta_W_v = self.linear_b_v.weight @ self.linear_a_v.weight
        weight_u_v, weight_sigma_v, weight_vt_v = self._cached_svd_v
        delta_sigma_v = torch.diag(torch.matmul(torch.matmul(weight_u_v.T, delta_W_v), weight_vt_v.T))
        _, top_index_v = self.calculate_change_rate(weight_sigma_v, delta_sigma_v, self.index)
        remain_index_v = torch.tensor([idx for idx in range(weight_u_v.shape[1])
                                       if idx not in top_index_v], device=device)
        new_base_W_v = (weight_u_v[:, remain_index_v] @ torch.diag(weight_sigma_v[remain_index_v])
                        @ weight_vt_v[remain_index_v, :])
        self.qkv.weight[-self.dim:, :] = new_base_W_v.clone()
        self._svd_cache_valid = False
        return top_index_q, top_index_v

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
    """Applies LoRA adaptation with EMA copies and subspace truncation to
    the SAM image encoder (repo LoRA_Sam)."""

    def __init__(self, sam_model, r: int, lora_layer=None, ema_mode=True,
                 Dash_index=8, top_s_change_rates=None, d_delta=False,
                 alpha: float = 0.1) -> None:
        super(LoRA_Sam, self).__init__()
        self.ema_mode = ema_mode
        self.top_s_change_rates = top_s_change_rates
        self.d_delta = d_delta
        # `alpha` (config): coefficient of the logarithmic L_tsd loss
        # `log(1 + alpha * change_rate)` (see `_LoRAtruncation_qkv.compute_l_tsd`).
        self.alpha = alpha

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
        # layers 4-8 from scratch) and the 8 CBA blocks are trained jointly
        # with the LoRA increments. Everything else in the ViT stays frozen.
        for name, param in sam_model.image_encoder.named_parameters():
            if name.startswith('cnn_encoder.') or '.cba' in name:
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

            self.index = Dash_index
            blk.attn.qkv = _LoRAtruncation_qkv(
                w_qkv_linear,
                w_a_linear_q, w_b_linear_q, w_a_linear_v, w_b_linear_v,
                ema_w_a_linear_q, ema_w_b_linear_q, ema_w_a_linear_v, ema_w_b_linear_v,
                index=self.index,
                top_s_change_rates=self.top_s_change_rates,
                d_delta=self.d_delta,
                alpha=self.alpha,
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
        (train/val/test/truncation) is forced onto the student LoRA.
        """
        return bool(ema) and bool(getattr(self, 'ema_mode', False))

    def compute_l_tsd(self, ema=False):
        """Compute total L_tsd (log-squashed rate without deltaW denominator).

        Aggregates `_LoRAtruncation_qkv.compute_l_tsd`, which selects the
        top-s directions with the original change rate but uses
        ``mean(log(1 + alpha * change_rate))`` as the loss value, where the
        rate is ``(bb / (a + eps)) ** 2`` (NO deltaW term in the
        denominator; ``a`` is the base-weight singular value sigma) and
        ``alpha`` comes from config (`self.alpha`, default 0.1).

        Convention: uses the student LoRA (ema=False); an `ema=True` request is
        honored only when the model owns EMA copies.
        """
        ema = self.resolve_ema(ema)
        total_l_tsd = torch.tensor(0.0, device=self.sam.device)
        count = 0
        for blk in self.sam.image_encoder.blocks:
            qkv = blk.attn.qkv
            if hasattr(qkv, 'compute_l_tsd'):
                l_tsd = qkv.compute_l_tsd(ema=ema)
                total_l_tsd = total_l_tsd + l_tsd
                count += 1
        return total_l_tsd / count if count > 0 else torch.tensor(0.0, device=self.sam.device)

    def cache_all_svd(self):
        """Cache SVD of base weights for all LoRA qkv modules (call once per epoch)."""
        for blk in self.sam.image_encoder.blocks:
            qkv = blk.attn.qkv
            if hasattr(qkv, 'cache_base_weight_svd'):
                qkv.cache_base_weight_svd()

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
               cnn_pretrained=True, cba_last_n=4, grassmann_detach=True,
               truncate_before_compute_grassmann=0, truncate_ref=False):
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
            grassmann_detach=grassmann_detach,
            truncate_before_compute_grassmann=truncate_before_compute_grassmann,
            truncate_ref=truncate_ref,
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
                 num_classes=1, rank=64, ema_mode=True, truncation_size=96,
                 top_s_change_rates=None, d_delta=False, alpha=0.1,
                 cnn_pretrained=True, cba_last_n=4,
                 grassmann_detach=True,
                 truncate_before_compute_grassmann=0, truncate_ref=False):
    """Build DSR-SAM2: SAM ViT + ResNet18 CNN branch + 8 CBA blocks, wrapped
    by the DSR-SAM LoRA/EMA/subspace-truncation wrapper.

    ``grassmann_detach``: True detaches G_score (fixed gate); the entry-point
    config exposes the inverse switch ``grassmann_grad`` (default False).

    ``truncate_before_compute_grassmann``: 0 keeps the usual DSR-SAM2 path.
    When > 0, the s smallest singular values/vectors of each branch feature
    are removed before G_score is computed. ``truncate_ref=True`` uses those
    truncated features only as the G_score reference and leaves the fused
    features untouched; ``truncate_ref=False`` fuses the truncated features.
    """
    assert model_type in VIT_CONFIGS
    sam, _ = _build_sam(image_size=image_size, num_classes=num_classes,
                        checkpoint=checkpoint, model_type=model_type,
                        cnn_pretrained=cnn_pretrained, cba_last_n=cba_last_n,
                        grassmann_detach=grassmann_detach,
                        truncate_before_compute_grassmann=truncate_before_compute_grassmann,
                        truncate_ref=truncate_ref)
    net = LoRA_Sam(sam, r=rank, ema_mode=ema_mode,
                   Dash_index=truncation_size,
                   top_s_change_rates=top_s_change_rates,
                   d_delta=d_delta,
                   alpha=alpha)
    return net
