"""
UltraSAM zero-shot evaluation (CAMMA-public/UltraSam).

The official UltraSam.pth is an mmdet training checkpoint (needs mmengine
to unpickle). Its modules map onto SAM-native components:
  * backbone (mmpretrain ViTSAM vit_b) -> standard SAM ViT-B encoder
    (patch_embed keeps its bias; channel_reduction is the 256-ch neck).
  * bbox_head (upscaling/hypernets/iou head) -> standard SAM mask decoder.
  * decoder (mmdet MultiheadAttention transformer) -> TwoWayTransformer
    with q/k/v/out biases (mapped from in_proj_bias splits).
  * prompt label table (11 x 256): rows CORNER_A/B + 4 MASK_OUT + IOU_OUT
    serve as token contents; box corners add standard positional
    encodings. Dense prompt = NON_INIT_MASK_EMBED row expanded.
Single-box zero-shot (no refinement cascade): tokens =
[corner_a, corner_b, mask x4, iou]; output = first (MASK_OUT) mask.

Runs at the native 1024 resolution with ImageNet normalization
(repo data_preprocessor); boxes are scaled from the 256-space
box_coords.json. Test-only: no training code.
"""
import math
from functools import partial
from typing import Any, List, Optional, Tuple, Type

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Label rows in prompt_encoder.label_embedding (repo EmbeddingIndex).
LBL_NON_INIT = 0
LBL_CORNER_A = 3
LBL_CORNER_B = 4
LBL_MASK_OUTS = (6, 7, 8, 9)
LBL_IOU_OUT = 10


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
# Image encoder (SAM ViT-B; patch embed keeps mmpretrain bias)
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
                 in_chans: int = 3, embed_dim: int = 768,
                 bias: bool = True) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=bias)

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
        self.window_size = window_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)
        x = self.norm1(x)
        x = self.attn(x)
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x


class ImageEncoderViT(nn.Module):
    def __init__(self, img_size: int = 1024, patch_size: int = 16,
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
            in_chans=in_chans, embed_dim=embed_dim, bias=True)
        self.pos_embed = None
        if use_abs_pos:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, 1024 // patch_size, 1024 // patch_size,
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            new_abs_pos = F.interpolate(
                self.pos_embed.permute(0, 3, 1, 2),
                size=(x.shape[1], x.shape[2]),
                mode='bicubic', align_corners=False).permute(0, 2, 3, 1)
            x = x + new_abs_pos
        for blk in self.blocks:
            x = blk(x)
        return self.neck(x.permute(0, 3, 1, 2))


# ============================================================
# Prompt module (label table + PE, repo prior_generators)
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


class UltraPrompt(nn.Module):
    """Box -> 7 sparse tokens + NON_INIT dense prompt (repo box path)."""

    def __init__(self, embed_dim: int = 256,
                 input_image_size: Tuple[int, int] = (1024, 1024),
                 image_embedding_size: Tuple[int, int] = (64, 64)) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.input_image_size = input_image_size
        self.image_embedding_size = image_embedding_size
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)
        self.label_embedding = nn.Embedding(11, embed_dim)

    def get_dense_pe(self) -> torch.Tensor:
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)

    def forward(self, boxes: torch.Tensor):
        # boxes [B,4] (x1,y1,x2,y2) -> corners [B,2,2].
        dev = self.label_embedding.weight.device
        boxes = boxes.to(dev)
        B = boxes.shape[0]
        corners = boxes.reshape(B, 2, 2)
        pos = self.pe_layer.forward_with_coords(corners,
                                                self.input_image_size)
        tok_a = self.label_embedding.weight[LBL_CORNER_A] + pos[:, 0, :]
        tok_b = self.label_embedding.weight[LBL_CORNER_B] + pos[:, 1, :]
        mask_toks = self.label_embedding.weight[
            list(LBL_MASK_OUTS), :].unsqueeze(0).expand(B, -1, -1)
        iou_tok = self.label_embedding.weight[
            LBL_IOU_OUT, :].unsqueeze(0).unsqueeze(0).expand(B, -1, -1)
        sparse = torch.cat([tok_a.unsqueeze(1), tok_b.unsqueeze(1),
                            mask_toks, iou_tok], dim=1)  # B,7,256
        dense = self.label_embedding.weight[
            LBL_NON_INIT, :].reshape(1, -1, 1, 1).expand(
                B, -1, *self.image_embedding_size)
        return sparse, dense


# ============================================================
# Two-way transformer with q/k/v/out biases (mmdet mapping)
# ============================================================

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


# ============================================================
# Mask decoder (bbox_head parts: upscaling/hypernets/iou head)
# ============================================================

class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_layers: int) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class MaskDecoder(nn.Module):
    def __init__(self, *, transformer_dim: int, transformer: nn.Module,
                 num_mask_tokens: int = 4,
                 activation: Type[nn.Module] = nn.GELU,
                 iou_head_depth: int = 3,
                 iou_head_hidden_dim: int = 256) -> None:
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer
        self.num_mask_tokens = num_mask_tokens
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
                tokens: torch.Tensor, dense: torch.Tensor):
        # tokens: [B,7,256] = [corner x2, mask x4, iou].
        B = image_embeddings.shape[0]
        src = image_embeddings + dense
        pos_src = (image_pe if image_pe.shape[0] == B
                   else image_pe.expand(B, -1, -1, -1))
        b, c, h, w = src.shape
        hs, src = self.transformer(src, pos_src, tokens)
        mask_tokens_out = hs[:, 2:6, :]
        iou_token_out = hs[:, 6:, :]
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)
        hyper_in_list = [self.output_hypernetworks_mlps[i](
            mask_tokens_out[:, i, :]) for i in range(self.num_mask_tokens)]
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(
            b, -1, h, w)
        iou_pred = self.iou_prediction_head(iou_token_out)
        # First mask = MASK_OUT token (non-ambiguous), like multimask=False.
        return masks[:, 0:1, :, :], iou_pred


# ============================================================
# UltraSAM wrapper (test-only)
# ============================================================

class UltraSAM(nn.Module):
    def __init__(self, image_encoder: ImageEncoderViT,
                 prompt: UltraPrompt, mask_decoder: MaskDecoder,
                 pixel_mean: List[float] = [0.485, 0.456, 0.406],
                 pixel_std: List[float] = [0.229, 0.224, 0.225]) -> None:
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt = prompt
        self.mask_decoder = mask_decoder
        self.register_buffer('pixel_mean',
                             torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer('pixel_std',
                             torch.Tensor(pixel_std).view(-1, 1, 1), False)

    @property
    def device(self) -> Any:
        return self.pixel_mean.device

    def forward(self, imgs1024: torch.Tensor, boxes1024: torch.Tensor):
        imge = self.image_encoder(self.preprocess(imgs1024))
        sparse, dense = self.prompt(boxes1024)
        image_pe = self.prompt.pe_layer(
            self.prompt.image_embedding_size).unsqueeze(0)
        masks, _ = self.mask_decoder(
            image_embeddings=imge, image_pe=image_pe,
            tokens=sparse, dense=dense)
        masks = self.postprocess_masks(
            masks, input_size=(1024, 1024), original_size=(1024, 1024))
        return {'masks': masks}

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
# Checkpoint conversion (mmdet -> native names) + builder
# ============================================================

def convert_ultrasam(raw):
    """Convert the mmdet state_dict to native module names.

    Returns (converted_dict, n_src_tensors). Raises on any unmapped key.
    """
    sd = raw['state_dict'] if isinstance(raw, dict) and 'state_dict' in raw \
        else raw
    out = {}
    n = [0]

    def put(dst, src):
        out[dst] = sd[src]
        n[0] += 1

    for k in sd:
        if k.startswith('backbone.'):
            r = k[len('backbone.'):]
            if r.startswith('patch_embed.projection.'):
                put('image_encoder.patch_embed.proj.' + r.split('.', 2)[2], k)
            elif r == 'pos_embed':
                put('image_encoder.pos_embed', k)
            elif r.startswith('channel_reduction.'):
                put('image_encoder.neck.' + r.split('.', 1)[1], k)
            elif r.startswith('layers.'):
                parts = r.split('.')
                rest = '.'.join(parts[2:])
                rest = rest.replace('ln1.', 'norm1.').replace(
                    'ln2.', 'norm2.')
                rest = rest.replace('ffn.layers.0.0.', 'mlp.lin1.').replace(
                    'ffn.layers.1.', 'mlp.lin2.')
                put(f'image_encoder.blocks.{parts[1]}.' + rest, k)
            else:
                raise KeyError(f'unmapped backbone key: {k}')
        elif k == 'prompt_encoder.label_encoder.label_embedding.weight':
            put('prompt.label_embedding.weight', k)
        elif k == ('prompt_encoder.pe_layer.'
                   'positional_encoding_gaussian_matrix'):
            put('prompt.pe_layer.positional_encoding_gaussian_matrix', k)
        elif k.startswith('prompt_encoder.mask_downscaling.'):
            # Only built with use_mask_refinement=True; unused here.
            n[0] += 1
            continue
        elif k.startswith('bbox_head.'):
            put(k.replace('bbox_head.', 'mask_decoder.'), k)
        elif k.startswith('decoder.layers.'):
            # decoder.layers.N.{self_attn,cross_*,mlp,norm*} (+ final below).
            parts = k.split('.')
            li, mod = parts[2], parts[3]
            rest = '.'.join(parts[4:])
            if rest.startswith('attn.'):
                rest = rest[len('attn.'):]
            base = f'mask_decoder.transformer.layers.{li}.'
            if mod == 'self_attn':
                base += 'self_attn.'
            elif mod == 'cross_attn_token_to_image':
                base += 'cross_attn_token_to_image.'
            elif mod == 'cross_attn_image_to_token':
                base += 'cross_attn_image_to_token.'
            elif mod == 'mlp':
                rest = rest.replace('layers.0.0.', 'lin1.').replace(
                    'layers.1.', 'lin2.')
                put(base + 'mlp.' + rest, k)
                continue
            elif mod in ('norm1', 'norm2', 'norm3', 'norm4'):
                put(base + mod + '.' + rest, k)
                continue
            else:
                raise KeyError(f'unmapped decoder key: {k}')
            if rest in ('q_proj_weight', 'k_proj_weight', 'v_proj_weight'):
                put(base + rest.replace('_weight', '.weight'), k)
            elif rest == 'in_proj_bias':
                d = sd[k].shape[0] // 3
                out[base + 'q_proj.bias'] = sd[k][:d]
                out[base + 'k_proj.bias'] = sd[k][d:2 * d]
                out[base + 'v_proj.bias'] = sd[k][2 * d:]
                n[0] += 3
            elif rest in ('out_proj.weight', 'out_proj.bias'):
                put(base + rest, k)
            else:
                raise KeyError(f'unmapped decoder key: {k}')
        elif k.startswith('decoder.final_attn_token_to_image.'):
            r = k[len('decoder.final_attn_token_to_image.'):]
            if r.startswith('attn.'):
                r = r[len('attn.'):]
            base = 'mask_decoder.transformer.final_attn_token_to_image.'
            if r in ('q_proj_weight', 'k_proj_weight', 'v_proj_weight'):
                put(base + r.replace('_weight', '.weight'), k)
            elif r == 'in_proj_bias':
                d = sd[k].shape[0] // 3
                out[base + 'q_proj.bias'] = sd[k][:d]
                out[base + 'k_proj.bias'] = sd[k][d:2 * d]
                out[base + 'v_proj.bias'] = sd[k][2 * d:]
                n[0] += 3
            elif r in ('out_proj.weight', 'out_proj.bias'):
                put(base + r, k)
            else:
                raise KeyError(f'unmapped decoder key: {k}')
        elif k == 'decoder.post_norm.weight' or k == 'decoder.post_norm.bias':
            put(k.replace('decoder.post_norm',
                          'mask_decoder.transformer.norm_final_attn'),
                k)
        else:
            raise KeyError(f'unmapped key: {k}')
    return out, n[0]


NATIVE_SIZE = 1024


def build_ultrasam(checkpoint=None, image_size=1024):
    """Build UltraSAM vit_b (test-only) and convert the checkpoint."""
    prompt_embed_dim = 256
    vit_patch_size = 16
    g = dict(encoder_embed_dim=768, encoder_depth=12, encoder_num_heads=12,
             encoder_global_attn_indexes=[2, 5, 8, 11])
    model = UltraSAM(
        image_encoder=ImageEncoderViT(
            depth=g['encoder_depth'], embed_dim=g['encoder_embed_dim'],
            img_size=image_size, mlp_ratio=4,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            num_heads=g['encoder_num_heads'], patch_size=vit_patch_size,
            qkv_bias=True, use_rel_pos=True,
            global_attn_indexes=g['encoder_global_attn_indexes'],
            window_size=14, out_chans=prompt_embed_dim),
        prompt=UltraPrompt(
            embed_dim=prompt_embed_dim,
            input_image_size=(image_size, image_size),
            image_embedding_size=(image_size // vit_patch_size,
                                  image_size // vit_patch_size)),
        mask_decoder=MaskDecoder(
            transformer_dim=prompt_embed_dim,
            transformer=TwoWayTransformer(
                depth=2, embedding_dim=prompt_embed_dim, mlp_dim=2048,
                num_heads=8)),
        pixel_mean=[0.485, 0.456, 0.406], pixel_std=[0.229, 0.224, 0.225],
    )
    model.eval()
    if checkpoint is not None:
        with open(checkpoint, 'rb') as f:
            try:
                raw = torch.load(f, map_location='cpu', weights_only=True)
            except Exception:
                # Raw MMDetection checkpoint (needs mmengine to unpickle).
                f.seek(0)
                raw = torch.load(f, map_location='cpu', weights_only=False)
        if isinstance(raw, dict) and 'state_dict' in raw:
            # Raw mmdet checkpoint (needs mmengine to unpickle); convert.
            conv, n_src = convert_ultrasam(raw)
            missing, unexpected = model.load_state_dict(conv, strict=False)
            print(f'UltraSAM: converted {n_src} tensors; '
                  f'missing={len(missing)} unexpected={len(unexpected)}')
            if missing:
                print('  missing:', missing)
        else:
            # Native converted checkpoint (weights/ultrasam/ultrasam_vit_b.pth).
            model.load_state_dict(raw)
            print(f'UltraSAM: loaded native checkpoint from {checkpoint}')
    return model
