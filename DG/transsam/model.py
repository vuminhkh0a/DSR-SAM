"""
Trans-SAM: Transfer Segment Anything Model to Medical Image Segmentation
with Parameter-Efficient Fine-Tuning (repo wuyanlin-wyl/Trans-SAM).

Architecture (repo SAM/modeling/):
  * IPF adapter (adapter_ipf.py): input image features are projected by
    adapter_sa and injected into every attention layer (deep_adapt),
    with TripletAttention + BatchNorm in the bottleneck.
  * MDT adapter (adapter_mdt.py): Convpass multi-scale convolutions
    (1x1 + 3x3 + 5x5, identity init) with BatchNorm + spatial attention,
    placed parallel to each block MLP (no residual, added to the path).
  * Prompt-free semantic segmentation: no prompts are used
    (points=None, boxes=None, masks=None), like sam_no_prompt.py.
  * Freezing (repo image_encoder.py): encoder frozen except 'adapter'
    params (covers adapter_sa/IPF/MDT); decoder + prompt encoder train.
"""
import math
from functools import partial
from typing import Any, List, Optional, Tuple, Type

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# IPF adapter (repo modeling/adapter_ipf.py)
# ============================================================

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, relu=True, bn=True,
                 bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation,
                              groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01,
                                 affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ZPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1),
                          torch.mean(x, 1).unsqueeze(1)), dim=1)


class AttentionGate(nn.Module):
    def __init__(self):
        super(AttentionGate, self).__init__()
        kernel_size = 7
        self.compress = ZPool()
        self.conv = BasicConv(2, 1, kernel_size, stride=1,
                              padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.conv(x_compress)
        scale = torch.sigmoid_(x_out)
        return x * scale


class TripletAttention(nn.Module):
    def __init__(self, no_spatial=False):
        super(TripletAttention, self).__init__()
        self.cw = AttentionGate()
        self.hc = AttentionGate()
        self.no_spatial = no_spatial
        if not no_spatial:
            self.hw = AttentionGate()

    def forward(self, x):
        x_perm1 = x.permute(0, 2, 1, 3).contiguous()
        x_out1 = self.cw(x_perm1)
        x_out11 = x_out1.permute(0, 2, 1, 3).contiguous()
        x_perm2 = x.permute(0, 3, 2, 1).contiguous()
        x_out2 = self.hc(x_perm2)
        x_out21 = x_out2.permute(0, 3, 2, 1).contiguous()
        if not self.no_spatial:
            x_out = self.hw(x)
            x_out = 1 / 3 * (x_out + x_out11 + x_out21)
        else:
            x_out = 1 / 2 * (x_out11 + x_out21)
        return x_out


class Adapter_IPF(nn.Module):
    def __init__(self, embedding_dim=96, mlp_dim=96, config=None,
                 d_model=None, bottleneck=None, dropout=0.0,
                 init_option='lora', adapter_scalar='1.0',
                 adapter_layernorm_option='in'):
        super().__init__()
        self.n_embd = embedding_dim
        self.mlp_dim = mlp_dim
        self.adapter_layernorm_option = adapter_layernorm_option
        self.adapter_layer_norm_before = None
        if adapter_layernorm_option == 'in' or adapter_layernorm_option == 'out':
            self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)
        if adapter_scalar == 'learnable_scalar':
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(adapter_scalar)
        self.down_proj = nn.Linear(self.n_embd, self.mlp_dim)
        self.non_linear_func = nn.ReLU(inplace=True)
        self.up_proj = nn.Linear(self.mlp_dim, self.n_embd)
        self.att2 = TripletAttention()
        self.bn = nn.BatchNorm2d(mlp_dim)
        self.dropout = dropout
        if init_option == 'bert':
            raise NotImplementedError
        elif init_option == 'lora':
            with torch.no_grad():
                nn.init.orthogonal_(self.down_proj.weight)
                nn.init.zeros_(self.up_proj.weight)
                nn.init.zeros_(self.down_proj.bias)
                nn.init.zeros_(self.up_proj.bias)

    def forward(self, x, deep_adapt=None, add_residual=True, residual=None):
        residual = x if residual is None else residual
        if self.adapter_layernorm_option == 'in':
            x = self.adapter_layer_norm_before(x)
        down = self.down_proj(x)
        down = self.non_linear_func(down)
        down = down + deep_adapt
        down = down.permute(0, 3, 1, 2)
        down = self.bn(down)
        down = self.att2(down)
        down = down.permute(0, 2, 3, 1)
        down = F.dropout(down, p=self.dropout, training=self.training)
        up = self.up_proj(down)
        up = up * self.scale
        if self.adapter_layernorm_option == 'out':
            up = self.adapter_layer_norm_before(up)
        if add_residual:
            output = up + residual
        else:
            output = up
        return output


# ============================================================
# MDT adapter (repo modeling/adapter_mdt.py)
# ============================================================

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class Convpass(nn.Module):
    def __init__(self, n_embd, dim=8, xavier_init=False):
        super().__init__()
        self.adapter_conv1 = nn.Conv2d(dim, dim, kernel_size=1, stride=1,
                                      padding=0)
        self.adapter_conv2 = nn.Conv2d(dim, dim, kernel_size=3, stride=1,
                                      padding=1)
        self.adapter_conv3 = nn.Conv2d(dim, dim, kernel_size=5, stride=1,
                                      padding=2)
        if xavier_init:
            nn.init.xavier_uniform_(self.adapter_conv1.weight)
        else:
            nn.init.zeros_(self.adapter_conv1.weight)
            nn.init.zeros_(self.adapter_conv2.weight)
            nn.init.zeros_(self.adapter_conv3.weight)
            self.adapter_conv1.weight.data[:, :, 0, 0] += torch.eye(
                dim, dtype=torch.float)
            self.adapter_conv2.weight.data[:, :, 1, 1] += torch.eye(
                dim, dtype=torch.float)
            self.adapter_conv3.weight.data[:, :, 2, 2] += torch.eye(
                dim, dtype=torch.float)
        nn.init.zeros_(self.adapter_conv1.bias)
        nn.init.zeros_(self.adapter_conv2.bias)
        nn.init.zeros_(self.adapter_conv3.bias)
        self.adapter_down = nn.Linear(n_embd, dim)
        self.adapter_up = nn.Linear(dim, n_embd)
        nn.init.xavier_uniform_(self.adapter_down.weight)
        nn.init.zeros_(self.adapter_down.bias)
        nn.init.zeros_(self.adapter_up.weight)
        nn.init.zeros_(self.adapter_up.bias)
        self.act = QuickGELU()
        self.dropout = nn.Dropout(0.1)
        self.dim = dim
        self.bn = nn.BatchNorm2d(dim)
        self.sa = SA()

    def forward(self, x):
        B, H, W, C = x.shape
        x = x.view(B, -1, C)
        x_down = self.adapter_down(x)
        x_down = self.act(x_down)
        x_patch = x_down.reshape(B, H, W, self.dim).permute(0, 3, 1, 2)
        x_patch1 = self.adapter_conv1(x_patch)
        x_patch2 = self.adapter_conv2(x_patch)
        x_patch3 = self.adapter_conv3(x_patch)
        x_patch = x_patch1 + x_patch2 + x_patch3
        x_patch = self.bn(x_patch)
        x_patch = x_patch.permute(0, 2, 3, 1).reshape(B, H * W, self.dim)
        x_down = self.act(x_patch)
        x_down = self.dropout(x_down)
        x_up = self.adapter_up(x_down)
        x_up = x_up.reshape(B, H, W, C).permute(0, 3, 1, 2)
        x_up = self.sa(x_up)
        x_up = x_up.permute(0, 2, 3, 1)
        return x_up


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        self.sa = _SpatialGate(kernel_size=kernel_size)

    def forward(self, x):
        residual = x
        out = x * self.sa(x)
        return out + residual


class _SpatialGate(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size,
                              padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], 1)
        output = self.conv(result)
        output = self.sigmoid(output)
        return output


class SA(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.sa = _SpatialGate(kernel_size=kernel_size)

    def forward(self, x):
        residual = x
        out = x * self.sa(x)
        return out + residual


class Adapter_MDT(nn.Module):
    def __init__(self, embedding_dim=96, mlp_dim=96, dropout=0.0,
                 init_option='lora', adapter_scalar='1.0',
                 adapter_layernorm_option='in'):
        super().__init__()
        self.n_embd = embedding_dim
        self.mlp_dim = mlp_dim
        self.adapter_layernorm_option = adapter_layernorm_option
        self.adapter_layer_norm_before = None
        if adapter_layernorm_option == 'in' or adapter_layernorm_option == 'out':
            self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)
        if adapter_scalar == 'learnable_scalar':
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(adapter_scalar)
        self.conv_pass = Convpass(self.n_embd, self.mlp_dim)
        self.dropout = dropout

    def forward(self, x, add_residual=True, residual=None):
        residual = x if residual is None else residual
        if self.adapter_layernorm_option == 'in':
            x = self.adapter_layer_norm_before(x)
        x = self.conv_pass(x)
        x = x * self.scale
        if self.adapter_layernorm_option == 'out':
            x = self.adapter_layer_norm_before(x)
        if add_residual:
            output = x + residual
        else:
            output = x
        return output


# ============================================================
# Common blocks (standard SAM)
# ============================================================

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
        self.adapter_deep = Adapter_IPF(dim, dim // 32)

    def forward(self, x: torch.Tensor, deep_adapt=None) -> torch.Tensor:
        B, H, W, C = x.shape
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
        x = self.proj(x)
        x = self.adapter_deep(x, deep_adapt)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, kernel_size: Tuple[int, int] = (16, 16),
                 stride: Tuple[int, int] = (16, 16),
                 padding: Tuple[int, int] = (0, 0),
                 in_chans: int = 3, embed_dim: int = 768) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size,
                              stride=stride, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return x.permute(0, 2, 3, 1)  # B C H W -> B H W C


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 qkv_bias: bool = True,
                 norm_layer: Type[nn.Module] = nn.LayerNorm,
                 act_layer: Type[nn.Module] = nn.GELU,
                 use_rel_pos: bool = False, rel_pos_zero_init: bool = True,
                 window_size: int = 0,
                 input_size: Optional[Tuple[int, int]] = None) -> None:
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
        self.adapter = Adapter_MDT(embedding_dim=dim, mlp_dim=dim // 8)
        self.window_size = window_size

    def forward(self, x: torch.Tensor, deep_adapt_sa=None) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)
            deep_adapt_sa, _ = window_partition(deep_adapt_sa,
                                                self.window_size)
        x = self.attn(x, deep_adapt_sa)
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))
        x = shortcut + x
        residual = x
        adapt_x = self.adapter(x, add_residual=False)
        x_mlp = self.mlp(self.norm2(x))
        x = x_mlp + adapt_x + residual
        return x


class ImageEncoderViT(nn.Module):
    def __init__(self, img_size: int = 256, patch_size: int = 16,
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
        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans, embed_dim=embed_dim)
        self.pos_embed = None
        if use_abs_pos:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, img_size // patch_size, img_size // patch_size,
                            embed_dim))
        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, norm_layer=norm_layer, act_layer=act_layer,
                use_rel_pos=use_rel_pos, rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i not in global_attn_indexes else 0,
                input_size=(img_size // patch_size, img_size // patch_size))
            self.blocks.append(block)
        self.neck = nn.Sequential(
            nn.Conv2d(embed_dim, out_chans, kernel_size=1, bias=False),
            LayerNorm2d(out_chans),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1,
                      bias=False),
            LayerNorm2d(out_chans),
        )
        self.adapter_sa = nn.Linear(embed_dim, embed_dim // 32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        deep_adapt_sa = self.adapter_sa(x)
        for blk in self.blocks:
            x = blk(x, deep_adapt_sa)
        return self.neck(x.permute(0, 3, 1, 2))


# ============================================================
# Prompt encoder (standard SAM; unused inputs, kept for parity)
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

    def forward(self, points, boxes, masks):
        if points is not None:
            bs = points[0].shape[0]
        elif boxes is not None:
            bs = boxes.shape[0]
        elif masks is not None:
            bs = masks.shape[0]
        else:
            bs = 1
        sparse_embeddings = torch.empty((bs, 0, self.embed_dim),
                                        device=self.point_embeddings[0].weight.device)
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
                 num_multimask_outputs: int = 1,
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
# Trans-SAM wrapper (prompt-free, repo sam_no_prompt.py)
# ============================================================

class TransSAM(nn.Module):
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

    @property
    def device(self) -> Any:
        return self.pixel_mean.device

    def forward(self, batched_input: torch.Tensor, multimask_output: bool,
                image_size: int):
        input_images = self.preprocess(batched_input)
        image_embeddings = self.image_encoder(input_images)
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=None, boxes=None, masks=None)
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output)
        masks = self.postprocess_masks(
            low_res_masks, input_size=(image_size, image_size),
            original_size=(image_size, image_size))
        return {'masks': masks, 'iou_predictions': iou_predictions,
                'low_res_logits': low_res_masks}

    def postprocess_masks(self, masks, input_size, original_size):
        masks = F.interpolate(
            masks, (self.image_encoder.img_size, self.image_encoder.img_size),
            mode='bilinear', align_corners=False)
        masks = masks[..., :input_size[0], :input_size[1]]
        masks = F.interpolate(masks, original_size, mode='bilinear',
                              align_corners=False)
        return masks

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.pixel_mean) / self.pixel_std
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        return F.pad(x, (0, padw, 0, padh))


# ============================================================
# Builder (vit_b/vit_h + y_DG-style SAM checkpoint loading)
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


def _load_from(model, state_dict, image_size, vit_patch_size,
               encoder_global_attn_indexes):
    ega = encoder_global_attn_indexes
    sam_dict = model.state_dict()
    new_state_dict = {k: v for k, v in state_dict.items()
                      if k in sam_dict.keys()
                      and v.shape == sam_dict[k].shape}
    token_size = int(image_size // vit_patch_size)
    ckpt_token = int(state_dict['image_encoder.pos_embed'].shape[1])
    if ckpt_token != token_size:
        pos_embed = state_dict['image_encoder.pos_embed']
        pos_embed = pos_embed.permute(0, 3, 1, 2)
        pos_embed = F.interpolate(pos_embed, (token_size, token_size),
                                  mode='bilinear', align_corners=False)
        pos_embed = pos_embed.permute(0, 2, 3, 1)
        new_state_dict['image_encoder.pos_embed'] = pos_embed
        rel_pos_keys = [k for k in sam_dict.keys() if 'rel_pos' in k]
        for rel_pos_key in rel_pos_keys:
            num = int(rel_pos_key.split('.')[2])
            if num in ega and rel_pos_key in state_dict:
                rel_pos_params = state_dict[rel_pos_key]
                h, w = rel_pos_params.shape
                rel_pos_params = rel_pos_params.unsqueeze(0).unsqueeze(0)
                rel_pos_params = F.interpolate(
                    rel_pos_params, (token_size * 2 - 1, w),
                    mode='bilinear', align_corners=False)
                new_state_dict[rel_pos_key] = rel_pos_params[0, 0, ...]
    sam_dict.update(new_state_dict)
    model.load_state_dict(sam_dict)


def build_transsam(checkpoint=None, model_type='vit_b', image_size=256,
                   num_classes=1):
    """Build Trans-SAM: encoder frozen except adapters (repo rule)."""
    assert model_type in VIT_CONFIGS
    cfg = VIT_CONFIGS[model_type]
    prompt_embed_dim = 256
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    model = TransSAM(
        image_encoder=ImageEncoderViT(
            depth=cfg['encoder_depth'], embed_dim=cfg['encoder_embed_dim'],
            img_size=image_size, mlp_ratio=4,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            num_heads=cfg['encoder_num_heads'], patch_size=vit_patch_size,
            qkv_bias=True, use_rel_pos=True,
            global_attn_indexes=cfg['encoder_global_attn_indexes'],
            window_size=14, out_chans=prompt_embed_dim),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size), mask_in_chans=16),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=num_classes,
            transformer=TwoWayTransformer(
                depth=2, embedding_dim=prompt_embed_dim, mlp_dim=2048,
                num_heads=8),
            transformer_dim=prompt_embed_dim),
        pixel_mean=[0.0, 0.0, 0.0], pixel_std=[1.0, 1.0, 1.0],
    )
    model.train()
    if checkpoint is not None:
        with open(checkpoint, 'rb') as f:
            state_dict = torch.load(f, map_location='cpu')
        _load_from(model, state_dict, image_size, vit_patch_size,
                   cfg['encoder_global_attn_indexes'])
        # Freeze the encoder except adapters (repo image_encoder.py);
        # prompt encoder + mask decoder stay trainable (repo default).
        for name, param in model.image_encoder.named_parameters():
            if 'adapter' not in name:
                param.requires_grad = False
    return model
