"""
DAPSAM: Prompting Segment Anything Model with Domain-Adaptive Prototype
for Generalizable Medical Image Segmentation (MICCAI 2024,
arXiv:2409.12522, repo wkklavis/DAPSAM, prostate/ branch).

Architecture (paper Sec. 2 / repo segment_anything/modeling/):
  * Generalized adapter: low-level features (LAdapter) are fused into the
    intermediate feature of every ViT block, filtered by channel attention
    (Filter.py, paper Eq. 1), then passed through a vanilla MLP adapter
    (paper Eq. 2, mlp_ratio 0.25, scale 0.5).
  * Prototype-based prompt generator: instance prototype (GAP+GMP, Eq. 3)
    queries a learnable memory bank (N=256, Eq. 4-5); the domain-adaptive
    prototype + cosine-similarity map + embedding are fused by a 1x1 conv
    into a dense prompt (Eq. 6-7). No manual prompts are used.
  * Mask decoder follows SAM (num_multimask_outputs = num_classes = 1).

Freezing (repo build_sam.py): image encoder frozen except adapters
('"adapter" not in name' rule covers LAdapter/channel_adapter);
mask decoder + prompt generator fully trainable.
"""
import math
from functools import partial
from typing import Any, List, Optional, Tuple, Type

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Attention filter (repo modeling/adapter/Filter.py, Eq. 1)
# ============================================================

class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.maxpool(x) + self.avgpool(x))


class ChannelFilter(nn.Module):
    def __init__(self, channel=768):
        super().__init__()
        self.ca = ChannelAttention(channel)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = x * self.ca(x)
        return x.permute(0, 2, 3, 1)


# ============================================================
# Adapters (repo modeling/adapter/Adapter.py + LAdapter.py)
# ============================================================

class Adapter(nn.Module):
    """Vanilla MLP adapter with channel filter (paper Eq. 1-2)."""

    def __init__(self, D_features, mlp_ratio=0.25, act_layer=nn.GELU,
                 skip_connect=True):
        super().__init__()
        self.channel_filter = ChannelFilter()
        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)
        self.act = act_layer()
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)

    def forward(self, x, embedding_feature):
        x = x + embedding_feature.permute(0, 2, 3, 1)
        x = self.channel_filter(x)
        xs = self.D_fc1(x)
        xs = self.act(xs)
        xs = self.D_fc2(xs)
        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return x


class LAdapter(nn.Module):
    """Low-level feature fusion adapter (paper Sec. 2.1)."""

    def __init__(self, scale_factor, embed_dim, depth):
        super(LAdapter, self).__init__()
        self.scale_factor = scale_factor
        self.embed_dim = embed_dim
        self.depth = depth
        self.embedding_generator = nn.Linear(self.embed_dim, self.embed_dim)
        for i in range(self.depth):
            lightweight_mlp = nn.Sequential(
                nn.Linear(self.embed_dim, self.embed_dim // self.scale_factor),
                nn.GELU(),
                nn.Linear(self.embed_dim // self.scale_factor, self.embed_dim),
            )
            channel_filter = ChannelFilter()
            setattr(self, 'lightweight_mlp_{}'.format(str(i)), lightweight_mlp)
            setattr(self, 'channel_filter_{}'.format(str(i)), channel_filter)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, mean=0., std=.02, a=-2., b=2.)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def init_embeddings(self, x):
        N, C, H, W = x.permute(0, 3, 1, 2).shape
        x = x.reshape(N, C, H * W).permute(0, 2, 1)
        embedding_feature = self.embedding_generator(x)
        return embedding_feature.permute(0, 2, 1).reshape(N, -1, H, W)

    def forward(self, i, feature, embedding_feature):
        N, C, H, W = embedding_feature.shape
        feature = feature.permute(0, 3, 1, 2)
        channel_filter = getattr(self, 'channel_filter_{}'.format(str(i)))
        filtered_feature = channel_filter(0.5 * feature + embedding_feature)
        filtered_feature = filtered_feature.reshape(N, C, H * W).permute(0, 2, 1)
        lightweight_mlp = getattr(self, 'lightweight_mlp_{}'.format(str(i)))
        return lightweight_mlp(filtered_feature)


# ============================================================
# Common blocks
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
    """ViT block with generalized adapter (repo image_encoder.py)."""

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
        self.window_size = window_size
        self.channel_adapter = Adapter(dim, skip_connect=False)
        self.scale = 0.5

    def forward(self, x: torch.Tensor, embedding_feature) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)
        x = self.attn(x)
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))
        x = shortcut + x
        xn = self.norm2(x)
        x = x + self.mlp(xn) + self.scale * self.channel_adapter(
            xn, embedding_feature)
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
        self.scale_factor = 4
        self.embed_dim = embed_dim
        self.l_adapter = LAdapter(self.scale_factor, self.embed_dim, depth)
        self.scale = 0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        N, H, W, C = x.shape
        if self.pos_embed is not None:
            x = x + self.pos_embed
        embedding_feature = self.l_adapter.init_embeddings(x)
        B = x.shape[0]
        for i, blk in enumerate(self.blocks):
            prompt = self.l_adapter(i, x, embedding_feature)
            x = prompt.reshape(B, H, W, -1) + x
            x = blk(x, embedding_feature)
        return self.neck(x.permute(0, 3, 1, 2))


# ============================================================
# Prototype prompt generator (repo modeling/memory/memory_prompt.py)
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


class MemoryUnit(nn.Module):
    """Learnable memory bank M in R^{N x C} (paper Eq. 4-5)."""

    def __init__(self, mem_dim=256, fea_dim=256):
        super(MemoryUnit, self).__init__()
        self.mem_dim = mem_dim
        self.fea_dim = fea_dim
        self.weight = nn.Parameter(torch.Tensor(self.mem_dim, self.fea_dim))
        self.bias = None
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        if len(input.shape) == 1:
            input = torch.unsqueeze(input, 0)
        att_weight = F.linear(input, self.weight)
        att_weight = F.softmax(att_weight, dim=1)
        mem_trans = self.weight.permute(1, 0)
        return F.linear(att_weight, mem_trans)


class PrototypePromptGenerate(nn.Module):
    """Domain-adaptive prompt generator (paper Eq. 3-7)."""

    def __init__(self, mem_dim=256, embed_dim=256, image_embedding_size=16):
        super(PrototypePromptGenerate, self).__init__()
        self.memory_bank = MemoryUnit(mem_dim, embed_dim)
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)
        self.image_embedding_size = (image_embedding_size, image_embedding_size)
        self.fuse_conv = nn.Conv2d(513, 256, 1)

    def get_dense_pe(self) -> torch.Tensor:
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)

    def forward(self, feature):
        N, C, H, W = feature.shape
        feature_proto_avg = F.avg_pool2d(input=feature,
                                         kernel_size=feature.shape[-2:])
        feature_proto_max = F.max_pool2d(input=feature,
                                         kernel_size=feature.shape[-2:])
        feature_proto = (feature_proto_avg + feature_proto_max).squeeze()
        di_proto = self.memory_bank(feature_proto)
        di_proto = di_proto.unsqueeze(2).unsqueeze(2)
        info_proto = di_proto.expand_as(feature)
        cos_sim_map = F.cosine_similarity(info_proto, feature, dim=1, eps=1e-7)
        cos_sim_map = cos_sim_map.unsqueeze(1)
        prompt = self.fuse_conv(
            torch.concat([feature, info_proto, cos_sim_map], dim=1))
        # Batch-correct empty sparse embeddings (repo hardcodes (1, 0, C)).
        sparse_embeddings = torch.empty((N, 0, C), device=feature.device)
        return sparse_embeddings, prompt


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
# Mask decoder (standard SAM, num_multimask_outputs = num_classes)
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
        # Batch-safe: repeat tokens/embeddings to the image batch size.
        B = image_embeddings.shape[0]
        if sparse_prompt_embeddings.shape[0] != B:
            sparse_prompt_embeddings = sparse_prompt_embeddings.expand(B, -1, -1)
        output_tokens = torch.cat([self.iou_token.weight,
                                   self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(B, -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)
        src = image_embeddings + dense_prompt_embeddings
        if image_pe.shape[0] != B:
            image_pe = image_pe.expand(B, -1, -1, -1)
        pos_src = image_pe
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
# DAPSAM wrapper (repo modeling/sam.py)
# ============================================================

class DAPSAM(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = 'RGB'

    def __init__(self, image_encoder: ImageEncoderViT,
                 mask_decoder: MaskDecoder, mem_dim: int = 256,
                 image_embedding_size: int = 16,
                 pixel_mean: List[float] = [0.0, 0.0, 0.0],
                 pixel_std: List[float] = [1.0, 1.0, 1.0]) -> None:
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_generator = PrototypePromptGenerate(
            mem_dim=mem_dim, embed_dim=256,
            image_embedding_size=image_embedding_size)
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
        sparse_embeddings, prompt = self.prompt_generator(image_embeddings)
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.prompt_generator.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=prompt,
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
# Builder (repo build_sam.py, vit_b/vit_h + y_DG-style loading)
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


def _load_from(sam, state_dict, image_size, vit_patch_size,
               encoder_global_attn_indexes):
    """Partial SAM load (repo load_from, generic global indexes).

    Decoder-only keys (mask_tokens/hypernetworks/iou head) are skipped
    because DAPSAM trains its own decoder; positional tables are
    interpolated when the token size differs.
    """
    ega = encoder_global_attn_indexes
    sam_dict = sam.state_dict()
    except_keys = ['mask_tokens', 'output_hypernetworks_mlps',
                   'iou_prediction_head']
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
            rel_pos_params = F.interpolate(
                rel_pos_params, (token_size * 2 - 1, w),
                mode='bilinear', align_corners=False)
            new_state_dict[k] = rel_pos_params[0, 0, ...]
    sam_dict.update(new_state_dict)
    sam.load_state_dict(sam_dict)


def build_dapsam(checkpoint=None, model_type='vit_b', image_size=256,
                 num_classes=1, mem_dim=256):
    """Build DAPSAM: encoder frozen except adapters (repo rule)."""
    assert model_type in VIT_CONFIGS
    cfg = VIT_CONFIGS[model_type]
    prompt_embed_dim = 256
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    model = DAPSAM(
        image_encoder=ImageEncoderViT(
            depth=cfg['encoder_depth'], embed_dim=cfg['encoder_embed_dim'],
            img_size=image_size, mlp_ratio=4,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            num_heads=cfg['encoder_num_heads'], patch_size=vit_patch_size,
            qkv_bias=True, use_rel_pos=True,
            global_attn_indexes=cfg['encoder_global_attn_indexes'],
            window_size=14, out_chans=prompt_embed_dim),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=num_classes,
            transformer=TwoWayTransformer(
                depth=2, embedding_dim=prompt_embed_dim, mlp_dim=2048,
                num_heads=8),
            transformer_dim=prompt_embed_dim),
        mem_dim=mem_dim, image_embedding_size=image_embedding_size,
        pixel_mean=[0.0, 0.0, 0.0], pixel_std=[1.0, 1.0, 1.0],
    )
    model.train()
    if checkpoint is not None:
        with open(checkpoint, 'rb') as f:
            state_dict = torch.load(f, map_location='cpu')
        _load_from(model, state_dict, image_size, vit_patch_size,
                   cfg['encoder_global_attn_indexes'])
        # Freeze the backbone except adapters (repo build_sam.py).
        for name, value in model.image_encoder.named_parameters():
            if 'adapter' not in name:
                value.requires_grad = False
    return model
