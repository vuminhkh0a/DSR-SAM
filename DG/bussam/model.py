"""
BUSSAM: Breast Ultrasound SAM Adapter (arXiv:2404.14837, repo
bscs12/BUSSAM, models/segment_anything_samus).

Differences vs SAMUS (same training recipe: point prompts, 0.2*BCE +
0.8*Dice with pos_weight 2, AdamW + warmup):
  * CNN branch (SingleCNNEmbed): BUS stem (64ch inc + 128ch stage) with
    Grouped_multi_axis_Hadamard_Product_Attention in the downsampling
    blocks (paper: lightweight CNN image encoder, local receptive field).
  * Cross-Branch Adapter: MLPAdapter fuses ViT + CNN features under
    spatial attention (repo common.py); applied on global blocks at
    depths 0 and 6, scale 0.5 (repo ParaBlock, kept verbatim).
  * Position Adapter (PostPosEmbed) + Feature Adapter (input_Adapter).

Freezing (repo modeling/samus.py): prompt encoder + mask decoder frozen;
image encoder frozen except cnn_embed / post_pos_embed / Adapter /
global-block rel_pos.
Only point prompts are used (repo forward ignores bbox).
"""
import math
from functools import partial
from typing import Any, List, Optional, Tuple, Type

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Common blocks (repo modeling/common.py)
# ============================================================

class Adapter(nn.Module):
    def __init__(self, D_features, mlp_ratio=0.25, act_layer=nn.GELU,
                 skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)
        self.act = act_layer()
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)

    def forward(self, x):
        xs = self.D_fc1(x)
        xs = self.act(xs)
        xs = self.D_fc2(xs)
        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return x


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7,
                              padding=7 // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        out = torch.cat([max_out, avg_out], dim=1)
        out = self.conv(out)
        out = self.sigmoid(out)
        return x * out


class MLPAdapter(nn.Module):
    """Cross-Branch Adapter: fuse ViT + CNN features under spatial
    attention, then MLP down/up (repo common.py)."""

    def __init__(self, D_features, mlp_ratio=0.25, act_layer=nn.GELU,
                 skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)
        self.act = act_layer()
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)
        self.spatial_attention = SpatialAttention()

    def forward(self, xn, cnnx):
        x = xn + cnnx
        B, H, W, C = x.size()
        xs = x.view(B, C, H, W)
        xs = self.spatial_attention(xs)
        xs = xs.view(B, H, W, C)
        xs = self.D_fc1(xs)
        xs = self.act(xs)
        xs = self.D_fc2(xs)
        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return x


class MLPBlock(nn.Module):
    def __init__(self, embedding_dim: int, mlp_dim: int,
                 act: Type[nn.Module] = nn.GELU) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lin1(x)
        x = self.act(x)
        x = self.lin2(x)
        return x


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


class _GHLayerNorm(nn.Module):
    """ConvNeXt-style LayerNorm with data_format (repo image_encoder.py)."""

    def __init__(self, normalized_shape, eps=1e-6,
                 data_format='channels_last'):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ['channels_last', 'channels_first']:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == 'channels_last':
            return F.layer_norm(x, self.normalized_shape, self.weight,
                                self.bias, self.eps)
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class SingleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                      padding=1, bias=False),
            LayerNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.conv(x)


class SingleDown(nn.Module):
    """Downscaling with GH attention (repo image_encoder.py)."""

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            Grouped_multi_axis_Hadamard_Product_Attention(in_channels,
                                                         out_channels),
            nn.GroupNorm(4, out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Grouped_multi_axis_Hadamard_Product_Attention(nn.Module):
    def __init__(self, dim_in, dim_out, x=8, y=8):
        super().__init__()
        c_dim_in = dim_in // 4
        k_size = 3
        pad = (k_size - 1) // 2
        self.params_xy = nn.Parameter(torch.Tensor(1, c_dim_in, x, y),
                                      requires_grad=True)
        nn.init.ones_(self.params_xy)
        self.conv_xy = nn.Sequential(
            nn.Conv2d(c_dim_in, c_dim_in, kernel_size=k_size, padding=pad,
                      groups=c_dim_in),
            nn.GELU(), nn.Conv2d(c_dim_in, c_dim_in, 1))
        self.params_zx = nn.Parameter(torch.Tensor(1, 1, c_dim_in, x),
                                      requires_grad=True)
        nn.init.ones_(self.params_zx)
        self.conv_zx = nn.Sequential(
            nn.Conv1d(c_dim_in, c_dim_in, kernel_size=k_size, padding=pad,
                      groups=c_dim_in),
            nn.GELU(), nn.Conv1d(c_dim_in, c_dim_in, 1))
        self.params_zy = nn.Parameter(torch.Tensor(1, 1, c_dim_in, y),
                                      requires_grad=True)
        nn.init.ones_(self.params_zy)
        self.conv_zy = nn.Sequential(
            nn.Conv1d(c_dim_in, c_dim_in, kernel_size=k_size, padding=pad,
                      groups=c_dim_in),
            nn.GELU(), nn.Conv1d(c_dim_in, c_dim_in, 1))
        self.dw = nn.Sequential(
            nn.Conv2d(c_dim_in, c_dim_in, 1),
            nn.GELU(),
            nn.Conv2d(c_dim_in, c_dim_in, kernel_size=3, padding=1,
                      groups=c_dim_in),
        )
        self.norm1 = _GHLayerNorm(dim_in, eps=1e-6,
                                  data_format='channels_first')
        self.norm2 = _GHLayerNorm(dim_in, eps=1e-6,
                                  data_format='channels_first')
        self.ldw = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size=3, padding=1,
                      groups=dim_in),
            nn.GELU(),
            nn.Conv2d(dim_in, dim_out, 1),
        )

    def forward(self, x):
        x = self.norm1(x)
        x1, x2, x3, x4 = torch.chunk(x, 4, dim=1)
        params_xy = self.params_xy
        x1 = x1 * self.conv_xy(F.interpolate(
            params_xy, size=x1.shape[2:4], mode='bilinear',
            align_corners=True))
        x2 = x2.permute(0, 3, 1, 2)
        params_zx = self.params_zx
        x2 = x2 * self.conv_zx(F.interpolate(
            params_zx, size=x2.shape[2:4], mode='bilinear',
            align_corners=True).squeeze(0)).unsqueeze(0)
        x2 = x2.permute(0, 2, 3, 1)
        x3 = x3.permute(0, 2, 1, 3)
        params_zy = self.params_zy
        x3 = x3 * self.conv_zy(F.interpolate(
            params_zy, size=x3.shape[2:4], mode='bilinear',
            align_corners=True).squeeze(0)).unsqueeze(0)
        x3 = x3.permute(0, 2, 1, 3)
        x4 = self.dw(x4)
        x = torch.cat([x1, x2, x3, x4], dim=1)
        x = self.norm2(x)
        x = self.ldw(x)
        return x


class SingleCNNEmbed(nn.Module):
    """BUS CNN stem (repo image_encoder.py)."""

    def __init__(self, patchsize: int = 8, in_chans: int = 3,
                 embed_dim: int = 768) -> None:
        super().__init__()
        downtimes = int(math.log2(patchsize))
        mid_channel = 128
        self.inc = SingleConv(in_chans, 64)
        self.encoder2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(64, mid_channel, 3, padding=1, bias=False),
            LayerNorm2d(mid_channel),
            nn.GELU(),
        )
        self.downs = nn.ModuleList()
        for i in range(downtimes - 1):
            if i == downtimes - 2:
                down = SingleDown(mid_channel, embed_dim)
            else:
                down = SingleDown(mid_channel, mid_channel * 2)
            mid_channel = mid_channel * 2
            self.downs.append(down)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.inc(x)
        x = self.encoder2(x)
        for down in self.downs:
            x = down(x)
        return x.permute(0, 2, 3, 1)  # B C H W -> B H W C


class PostPosEmbed(nn.Module):
    """Position adapter: downsample SAM pos_embed to the new feature size."""

    def __init__(self, embed_dim: int = 768, ori_feature_size: int = 64,
                 new_feature_size: int = 32) -> None:
        super().__init__()
        downtimes = int(math.log2(ori_feature_size // new_feature_size))
        self.downs = nn.ModuleList()
        for _ in range(downtimes):
            self.downs.append(SingleDown(embed_dim, embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 3, 1, 2)  # B H W C -> B C H W
        for down in self.downs:
            x = down(x)
        return x.permute(0, 2, 3, 1)  # B C H W -> B H W C


class PatchEmbed0(nn.Module):
    """ViT patch embedding used by BUSSAM (repo image_encoder.py)."""

    def __init__(self, kernel_size: Tuple[int, int] = (16, 16),
                 stride: Tuple[int, int] = (8, 8),
                 padding: Tuple[int, int] = (0, 0),
                 in_chans: int = 3, embed_dim: int = 768) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=16,
                              stride=(8, 8), padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, (256 + 8, 256 + 8), mode='bilinear',
                          align_corners=False)
        x = self.proj(x)
        return x.permute(0, 2, 3, 1)  # B C H W -> B H W C


# ============================================================
# Image encoder (repo modeling/image_encoder.py)
# ============================================================

def window_partition(x: torch.Tensor, window_size: int):
    B, H, W, C = x.shape
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w
    x = x.view(B, Hp // window_size, window_size, Wp // window_size,
               window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(
        -1, window_size, window_size, C)
    return windows, (Hp, Wp)


def window_unpartition(windows: torch.Tensor, window_size: int,
                       pad_hw: Tuple[int, int], hw: Tuple[int, int]):
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size,
                     window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)
    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x


def get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor):
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    if rel_pos.shape[0] != max_rel_dist:
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist, mode='linear')
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = ((q_coords - k_coords)
                       + (k_size - 1) * max(q_size / k_size, 1.0))
    return rel_pos_resized[relative_coords.long()]


def add_decomposed_rel_pos(attn: torch.Tensor, q: torch.Tensor,
                           rel_pos_h: torch.Tensor, rel_pos_w: torch.Tensor,
                           q_size: Tuple[int, int], k_size: Tuple[int, int]):
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)
    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = torch.einsum('bhwc,hkc->bhwk', r_q, Rh)
    rel_w = torch.einsum('bhwc,wkc->bhwk', r_q, Rw)
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
            assert input_size is not None
            self.rel_pos_h = nn.Parameter(
                torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(
                torch.zeros(2 * input_size[1] - 1, head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(
            2, 0, 3, 1, 4)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)
        attn = (q * self.scale) @ k.transpose(-2, -1)
        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h,
                                          self.rel_pos_w, (H, W), (H, W))
        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(
            0, 2, 3, 1, 4).reshape(B, H, W, -1)
        return self.proj(x)


class ParaBlock(nn.Module):
    """ViT block with Cross-Branch Adapters on global blocks at depths
    0 and 6 (repo image_encoder.py, kept verbatim)."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 qkv_bias: bool = True,
                 norm_layer: Type[nn.Module] = nn.LayerNorm,
                 act_layer: Type[nn.Module] = nn.GELU,
                 use_rel_pos: bool = False, rel_pos_zero_init: bool = True,
                 window_size: int = 0,
                 input_size: Optional[Tuple[int, int]] = None,
                 depth: int = 0) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos, rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size,
                                                            window_size))
        self.norm2 = norm_layer(dim)
        self.mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio),
                            act=act_layer)
        self.window_size = window_size
        if self.window_size == 0:
            self.MLP_Adapter1 = MLPAdapter(dim, skip_connect=False)
            self.MLP_Adapter = MLPAdapter(dim, skip_connect=False)
            self.refine_Adapter = SingleConv(in_channels=dim,
                                             out_channels=dim)
            self.scale = 0.5
        self.dim = dim
        self.depth = depth

    def forward(self, x: torch.Tensor, cnnx: torch.Tensor):
        shortcut = x
        x = self.norm1(x)
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)
        if self.window_size == 0 and (self.depth == 0 or self.depth == 6):
            sax = self.MLP_Adapter1(x, cnnx)
            x = x + sax
            cnnx = self.refine_Adapter(
                cnnx.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        x = self.attn(x)
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))
        x = shortcut + x
        xn = self.norm2(x)
        x = x + self.mlp(xn)
        if self.window_size == 0 and (self.depth == 0 or self.depth == 6):
            x = x + self.scale * self.MLP_Adapter(xn, cnnx)
        return x, cnnx


class ImageEncoderViT(nn.Module):
    def __init__(self, img_size: int = 256, patch_size: int = 8,
                 in_chans: int = 3, embed_dim: int = 768, depth: int = 12,
                 num_heads: int = 12, mlp_ratio: float = 4.0,
                 out_chans: int = 256, qkv_bias: bool = True,
                 norm_layer: Type[nn.Module] = nn.LayerNorm,
                 act_layer: Type[nn.Module] = nn.GELU,
                 use_abs_pos: bool = True, use_rel_pos: bool = False,
                 rel_pos_zero_init: bool = True, window_size: int = 0,
                 global_attn_indexes: Tuple[int, ...] = ()) -> None:
        super().__init__()
        self.img_size = img_size
        self.cnn_embed = SingleCNNEmbed(patchsize=patch_size, in_chans=3,
                                       embed_dim=embed_dim)
        self.patch_embed = PatchEmbed0(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans, embed_dim=embed_dim)
        self.pos_embed = None
        if use_abs_pos:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, 1024 // 16, 1024 // 16, embed_dim))
            self.post_pos_embed = PostPosEmbed(
                embed_dim=embed_dim, ori_feature_size=1024 // 16,
                new_feature_size=img_size // patch_size)
        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = ParaBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, norm_layer=norm_layer, act_layer=act_layer,
                use_rel_pos=use_rel_pos, rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i not in global_attn_indexes else 0,
                input_size=(img_size // patch_size, img_size // patch_size),
                depth=i)
            self.blocks.append(block)
        self.neck = nn.Sequential(
            nn.Conv2d(embed_dim, out_chans, kernel_size=1, bias=False),
            LayerNorm2d(out_chans),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1,
                      bias=False),
            LayerNorm2d(out_chans),
        )
        self.input_Adapter = Adapter(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        cnnx = self.cnn_embed(x)
        x = self.patch_embed(x)
        x = self.input_Adapter(x)
        if self.pos_embed is not None:
            pos_embed = self.post_pos_embed(self.pos_embed)
            x = x + pos_embed.repeat(x.shape[0], 1, 1, 1)
        for blk in self.blocks:
            x, cnnx = blk(x, cnnx)
        x = x + 0.5 * cnnx
        return self.neck(x.permute(0, 3, 1, 2))


# ============================================================
# Prompt encoder (standard SAM, click points)
# ============================================================

class PositionEmbeddingRandom(nn.Module):
    def __init__(self, num_pos_feats: int = 64,
                 scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer('positional_encoding_gaussian_matrix',
                             scale * torch.randn((2, num_pos_feats)))

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        h, w = size
        device = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
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
        return self._pe_encoding(coords.to(torch.float32))


class PromptEncoder(nn.Module):
    def __init__(self, embed_dim: int,
                 image_embedding_size: Tuple[int, int],
                 input_image_size: Tuple[int, int], mask_in_chans: int,
                 activation: Type[nn.Module] = nn.GELU) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.input_image_size = input_image_size
        self.image_embedding_size = image_embedding_size
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)
        self.num_point_embeddings: int = 4
        point_embeddings = [nn.Embedding(1, embed_dim)
                            for _ in range(self.num_point_embeddings)]
        self.point_embeddings = nn.ModuleList(point_embeddings)
        self.not_a_point_embed = nn.Embedding(1, embed_dim)
        self.mask_input_size = (4 * image_embedding_size[0],
                                4 * image_embedding_size[1])
        self.mask_downscaling = nn.Sequential(
            nn.Conv2d(1, mask_in_chans // 4, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans // 4),
            activation(),
            nn.Conv2d(mask_in_chans // 4, mask_in_chans, kernel_size=2,
                      stride=2),
            LayerNorm2d(mask_in_chans),
            activation(),
            nn.Conv2d(mask_in_chans, embed_dim, kernel_size=1),
        )
        self.no_mask_embed = nn.Embedding(1, embed_dim)

    def get_dense_pe(self) -> torch.Tensor:
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)

    def _embed_points(self, points: torch.Tensor, labels: torch.Tensor,
                      pad: bool) -> torch.Tensor:
        points = points + 0.5
        if pad:
            padding_point = torch.zeros((points.shape[0], 1, 2),
                                        device=points.device)
            padding_label = -torch.ones((labels.shape[0], 1),
                                        device=labels.device)
            points = torch.cat([points, padding_point], dim=1)
            labels = torch.cat([labels, padding_label], dim=1)
        point_embedding = self.pe_layer.forward_with_coords(
            points, self.input_image_size)
        point_embedding[labels == -1] = 0.0
        point_embedding[labels == -1] += self.not_a_point_embed.weight
        point_embedding[labels == 0] += self.point_embeddings[0].weight
        point_embedding[labels == 1] += self.point_embeddings[1].weight
        return point_embedding

    def forward(self, points, boxes, masks):
        bs = points[0].shape[0] if points is not None else 1
        sparse_embeddings = torch.empty((bs, 0, self.embed_dim),
                                        device=self.point_embeddings[0].weight.device)
        if points is not None:
            coords, labels = points
            point_embeddings = self._embed_points(coords, labels, pad=True)
            sparse_embeddings = torch.cat([sparse_embeddings,
                                           point_embeddings], dim=1)
        if masks is not None:
            dense_embeddings = self.mask_downscaling(masks)
        else:
            dense_embeddings = self.no_mask_embed.weight.reshape(
                1, -1, 1, 1).expand(bs, -1, self.image_embedding_size[0],
                                    self.image_embedding_size[1])
        return sparse_embeddings, dense_embeddings


# ============================================================
# Two-way transformer (standard SAM)
# ============================================================

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
                embedding_dim=embedding_dim, num_heads=num_heads,
                mlp_dim=mlp_dim, activation=activation,
                attention_downsample_rate=attention_downsample_rate,
                skip_first_layer_pe=(i == 0)))
        self.final_attn_token_to_image = AttentionT(
            embedding_dim, num_heads,
            downsample_rate=attention_downsample_rate)
        self.norm_final_attn = nn.LayerNorm(embedding_dim)

    def forward(self, image_embedding: torch.Tensor, image_pe: torch.Tensor,
                point_embedding: torch.Tensor):
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


class TwoWayAttentionBlock(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int,
                 mlp_dim: int = 2048,
                 activation: Type[nn.Module] = nn.ReLU,
                 attention_downsample_rate: int = 2,
                 skip_first_layer_pe: bool = False) -> None:
        super().__init__()
        self.self_attn = AttentionT(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.cross_attn_token_to_image = AttentionT(
            embedding_dim, num_heads,
            downsample_rate=attention_downsample_rate)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.mlp = MLPBlock(embedding_dim, mlp_dim, activation)
        self.norm3 = nn.LayerNorm(embedding_dim)
        self.norm4 = nn.LayerNorm(embedding_dim)
        self.cross_attn_image_to_token = AttentionT(
            embedding_dim, num_heads,
            downsample_rate=attention_downsample_rate)
        self.skip_first_layer_pe = skip_first_layer_pe

    def forward(self, queries: torch.Tensor, keys: torch.Tensor,
                query_pe: torch.Tensor, key_pe: torch.Tensor):
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


class AttentionT(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int,
                 downsample_rate: int = 1) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0
        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    def _separate_heads(self, x: torch.Tensor, num_heads: int):
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)

    def _recombine_heads(self, x: torch.Tensor):
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
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


# ============================================================
# Mask decoder (standard SAM)
# ============================================================

class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_layers: int, sigmoid_output: bool = False) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x


class MaskDecoder(nn.Module):
    def __init__(self, *, transformer_dim: int, transformer: nn.Module,
                 num_multimask_outputs: int = 3,
                 activation: Type[nn.Module] = nn.GELU,
                 iou_head_depth: int = 3,
                 iou_head_hidden_dim: int = 256) -> None:
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer
        self.num_multimask_outputs = num_multimask_outputs
        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4,
                               kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8,
                               kernel_size=2, stride=2),
            activation(),
        )
        self.output_hypernetworks_mlps = nn.ModuleList([
            MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
            for _ in range(self.num_mask_tokens)])
        self.iou_prediction_head = MLP(transformer_dim, iou_head_hidden_dim,
                                       self.num_mask_tokens, iou_head_depth)

    def forward(self, image_embeddings: torch.Tensor, image_pe: torch.Tensor,
                sparse_prompt_embeddings: torch.Tensor,
                dense_prompt_embeddings: torch.Tensor,
                multimask_output: bool):
        masks, iou_pred = self.predict_masks(
            image_embeddings=image_embeddings, image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings)
        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)
        masks = masks[:, mask_slice, :, :]
        iou_pred = iou_pred[:, mask_slice]
        return masks, iou_pred

    def predict_masks(self, image_embeddings: torch.Tensor,
                      image_pe: torch.Tensor,
                      sparse_prompt_embeddings: torch.Tensor,
                      dense_prompt_embeddings: torch.Tensor):
        B = image_embeddings.shape[0]
        if sparse_prompt_embeddings.shape[0] != B:
            sparse_prompt_embeddings = sparse_prompt_embeddings.expand(B, -1, -1)
        output_tokens = torch.cat([self.iou_token.weight,
                                   self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(B, -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)
        src = image_embeddings + dense_prompt_embeddings
        pos_src = (image_pe if image_pe.shape[0] == B
                   else image_pe.expand(B, -1, -1, -1))
        b, c, h, w = src.shape
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1:(1 + self.num_mask_tokens), :]
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)
        hyper_in_list = [self.output_hypernetworks_mlps[i](
            mask_tokens_out[:, i, :]) for i in range(self.num_mask_tokens)]
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(
            b, -1, h, w)
        iou_pred = self.iou_prediction_head(iou_token_out)
        return masks, iou_pred


# ============================================================
# Bussam wrapper (repo modeling/samus.py)
# ============================================================

class Bussam(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = 'RGB'

    def __init__(self, image_encoder: ImageEncoderViT,
                 prompt_encoder: PromptEncoder, mask_decoder: MaskDecoder,
                 pixel_mean: List[float] = [0.0, 0.0, 0.0],
                 pixel_std: List[float] = [1.0, 1.0, 1.0]) -> None:
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.register_buffer('pixel_mean',
                             torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer('pixel_std',
                             torch.Tensor(pixel_std).view(-1, 1, 1), False)

        for param in self.prompt_encoder.parameters():
            param.requires_grad = False
        for param in self.mask_decoder.parameters():
            param.requires_grad = False
        for n, value in self.image_encoder.named_parameters():
            if ('cnn_embed' not in n and 'post_pos_embed' not in n
                    and 'Adapter' not in n
                    and '2.attn.rel_pos' not in n
                    and '5.attn.rel_pos' not in n
                    and '8.attn.rel_pos' not in n
                    and '11.attn.rel_pos' not in n
                    and 'upneck' not in n):
                value.requires_grad = False

    @property
    def device(self) -> Any:
        return self.pixel_mean.device

    def forward(self, imgs: torch.Tensor,
                pt: Tuple[torch.Tensor, torch.Tensor],
                bbox: Optional[torch.Tensor] = None):
        """Point-prompt forward (repo: bbox arg accepted but unused)."""
        imge = self.image_encoder(imgs)
        if len(pt[0].shape) == 3:
            se, de = self.prompt_encoder(points=pt, boxes=None, masks=None)
            low_res_masks, _ = self.mask_decoder(
                image_embeddings=imge,
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=se,
                dense_prompt_embeddings=de,
                multimask_output=False)
            masks = F.interpolate(low_res_masks, (256, 256), mode='bilinear',
                                  align_corners=False)
            return {'low_res_logits': low_res_masks, 'masks': masks}
        low_res_masks, masks = [], []
        for i in range(pt[0].shape[1]):
            pti = (pt[0][:, i, :, :], pt[1][:, i, :])
            sei, dei = self.prompt_encoder(points=pti, boxes=None, masks=None)
            low_res_masksi, _ = self.mask_decoder(
                image_embeddings=imge,
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sei,
                dense_prompt_embeddings=dei,
                multimask_output=False)
            masksi = F.interpolate(low_res_masksi, (256, 256),
                                   mode='bilinear', align_corners=False)
            low_res_masks.append(low_res_masksi)
            masks.append(masksi)
        low_res_masks = torch.stack(low_res_masks, dim=1)
        masks = torch.stack(masks, dim=1)
        masks = masks.reshape(masks.shape[0], -1, masks.shape[3],
                              masks.shape[4])
        low_res_masks = low_res_masks.reshape(
            low_res_masks.shape[0], -1, low_res_masks.shape[3],
            low_res_masks.shape[4])
        return {'low_res_logits': low_res_masks, 'masks': masks}


# ============================================================
# Builder (repo build_sam_us.py, vit_b/vit_h + y_DG-style loading)
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


def _load_from(model, state_dict, image_size, patch_size):
    """Partial SAM load (repo load_from2): matching shapes copied, global
    rel_pos interpolated, BUSSAM-only params keep their init."""
    sam_dict = model.state_dict()
    clean = {}
    for k, v in state_dict.items():
        k = k[7:] if k.startswith('module.') else k
        if k in sam_dict and sam_dict[k].shape == v.shape:
            clean[k] = v
    token_size = int(image_size // patch_size)
    for k, v in state_dict.items():
        k = k[7:] if k.startswith('module.') else k
        if ('rel_pos' in k and k in sam_dict and k not in clean
                and sam_dict[k].dim() == 2 and v.dim() == 2
                and sam_dict[k].shape[1] == v.shape[1]):
            h, w = v.shape
            resized = F.interpolate(v.unsqueeze(0).unsqueeze(0),
                                    (token_size * 2 - 1, w),
                                    mode='bilinear', align_corners=False)
            clean[k] = resized[0, 0, ...]
    if 'image_encoder.pos_embed' in sam_dict:
        pos = state_dict.get('image_encoder.pos_embed',
                             state_dict.get('module.image_encoder.pos_embed'))
        if pos is not None and pos.shape == sam_dict[
                'image_encoder.pos_embed'].shape:
            clean['image_encoder.pos_embed'] = pos
    sam_dict.update(clean)
    model.load_state_dict(sam_dict)
    return len(clean)


def build_bussam(checkpoint=None, model_type='vit_b', image_size=256):
    """Build BUSSAM (frozen PEFT wrapper around the BUS encoder)."""
    assert model_type in VIT_CONFIGS
    cfg = VIT_CONFIGS[model_type]
    prompt_embed_dim = 256
    patch_size = image_size // 32
    image_embedding_size = image_size // patch_size
    model = Bussam(
        image_encoder=ImageEncoderViT(
            depth=cfg['encoder_depth'], embed_dim=cfg['encoder_embed_dim'],
            img_size=image_size, mlp_ratio=4,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            num_heads=cfg['encoder_num_heads'], patch_size=patch_size,
            qkv_bias=True, use_rel_pos=True,
            global_attn_indexes=cfg['encoder_global_attn_indexes'],
            window_size=14, out_chans=prompt_embed_dim),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size), mask_in_chans=16),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2, embedding_dim=prompt_embed_dim, mlp_dim=2048,
                num_heads=8),
            transformer_dim=prompt_embed_dim),
        pixel_mean=[0.0, 0.0, 0.0],
        pixel_std=[1.0, 1.0, 1.0],
    )
    model.train()
    if checkpoint is not None:
        with open(checkpoint, 'rb') as f:
            state_dict = torch.load(f, map_location='cpu')
        n = _load_from(model, state_dict, image_size, patch_size)
        print(f'BUSSAM: loaded {n} params from {checkpoint}')
    return model
