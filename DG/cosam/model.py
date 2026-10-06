"""
CoSAM: Self-Correcting SAM for Domain Generalization in 2D Medical Image
Segmentation (arXiv:2411.10136).

No public code was provided with the request, so this is implemented from
the paper description (Alg. 1/2, Eq. 1-9, Sec. 5.2), reusing the standard
SAM building blocks shared by the other y_DG DG methods.

Architecture (paper Sec. 4):
  * Frozen SAM image encoder + learnable mask decoder. The coarse mask is
    produced prompt-free: y~ = M(E(x), empty, empty) (Eq. 1).
  * Mask encoder = frozen dense prompt encoder Pd. The error decoder E
    (U-Net decoder) takes concat[E(x), Pd(yp)] and predicts the error map
    (Eq. 4); the error label is XOR(bin(y~), y) (correct=0, error=1).
  * Refined/guided masks use point (top-K error / random GT), box
    (largest-CC bounding box) and mask prompts (Eq. 7-8).
"""
import math
from functools import partial
from typing import Any, List, Optional, Tuple, Type

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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
# Image encoder (standard SAM ViT, fully frozen in CoSAM)
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
        x = self.norm1(x)
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)
        x = self.attn(x)
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        return self.neck(x.permute(0, 3, 1, 2))


# ============================================================
# Prompt encoder (standard SAM, sparse + dense)
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

    def _embed_boxes(self, boxes: torch.Tensor) -> torch.Tensor:
        boxes = boxes + 0.5
        coords = boxes.reshape(-1, 2, 2)
        corner_embedding = self.pe_layer.forward_with_coords(
            coords, self.input_image_size)
        corner_embedding[:, 0, :] += self.point_embeddings[2].weight
        corner_embedding[:, 1, :] += self.point_embeddings[3].weight
        return corner_embedding

    def _get_batch_size(self, points, boxes, masks) -> int:
        if points is not None:
            return points[0].shape[0]
        elif boxes is not None:
            return boxes.shape[0]
        elif masks is not None:
            return masks.shape[0]
        return 1

    def _get_device(self):
        return self.point_embeddings[0].weight.device

    def forward(self, points, boxes, masks):
        bs = self._get_batch_size(points, boxes, masks)
        sparse_embeddings = torch.empty((bs, 0, self.embed_dim),
                                        device=self._get_device())
        if points is not None:
            coords, labels = points
            point_embeddings = self._embed_points(coords, labels,
                                                  pad=(boxes is None))
            sparse_embeddings = torch.cat([sparse_embeddings,
                                           point_embeddings], dim=1)
        if boxes is not None:
            box_embeddings = self._embed_boxes(boxes)
            sparse_embeddings = torch.cat([sparse_embeddings,
                                           box_embeddings], dim=1)
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
# Error decoder (paper Sec. 4.3: U-Net decoder)
# ============================================================

class _DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class ErrorDecoder(nn.Module):
    """U-Net style decoder: concat[E(x), Pd(mask)] (512ch @ 16x16 for a
    256px input) -> full-resolution single-channel error map."""

    def __init__(self, in_chans=512):
        super().__init__()
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear',
                               align_corners=False)
        self.conv1 = _DoubleConv(in_chans, 256)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear',
                               align_corners=False)
        self.conv2 = _DoubleConv(256, 128)
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear',
                               align_corners=False)
        self.conv3 = _DoubleConv(128, 64)
        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear',
                               align_corners=False)
        self.conv4 = _DoubleConv(64, 32)
        self.out = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        x = self.conv1(self.up1(x))
        x = self.conv2(self.up2(x))
        x = self.conv3(self.up3(x))
        x = self.conv4(self.up4(x))
        return self.out(x)


# ============================================================
# CoSAM wrapper
# ============================================================

class CoSAM(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = 'RGB'

    def __init__(self, image_encoder: ImageEncoderViT,
                 prompt_encoder: PromptEncoder, mask_decoder: MaskDecoder,
                 error_decoder: ErrorDecoder,
                 pixel_mean: List[float] = [0.0, 0.0, 0.0],
                 pixel_std: List[float] = [1.0, 1.0, 1.0]) -> None:
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.error_decoder = error_decoder
        self.register_buffer('pixel_mean',
                             torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer('pixel_std',
                             torch.Tensor(pixel_std).view(-1, 1, 1), False)

    @property
    def device(self) -> Any:
        return self.pixel_mean.device

    @property
    def mask_prompt_size(self) -> Tuple[int, int]:
        h, w = self.prompt_encoder.mask_input_size
        return (h, w)

    def encode_image(self, x: torch.Tensor) -> torch.Tensor:
        return self.image_encoder(self.preprocess(x))

    def mask_embedding(self, bin_mask: torch.Tensor) -> torch.Tensor:
        """Frozen dense prompt encoder used as the mask encoder."""
        h, w = self.mask_prompt_size
        if bin_mask.shape[-2:] != (h, w):
            bin_mask = F.interpolate(bin_mask.float(), size=(h, w),
                                     mode='nearest')
        _, dense = self.prompt_encoder(points=None, boxes=None,
                                       masks=bin_mask.float())
        return dense

    def decode(self, image_emb: torch.Tensor, points, boxes,
               mask_prompt: Optional[torch.Tensor]):
        """Mask decoder with optional point/box/mask prompts."""
        if mask_prompt is not None:
            h, w = self.mask_prompt_size
            if mask_prompt.shape[-2:] != (h, w):
                mask_prompt = F.interpolate(mask_prompt.float(), size=(h, w),
                                            mode='nearest')
            _, dense = self.prompt_encoder(points=None, boxes=None,
                                           masks=mask_prompt.float())
        else:
            B = image_emb.shape[0]
            dense = self.prompt_encoder.no_mask_embed.weight.reshape(
                1, -1, 1, 1).expand(
                    B, -1, *self.prompt_encoder.image_embedding_size)
        if points is None and boxes is None:
            B = image_emb.shape[0]
            sparse = torch.empty(
                (B, 0, self.prompt_encoder.embed_dim), device=image_emb.device)
        else:
            sparse, _ = self.prompt_encoder(points=points, boxes=boxes,
                                            masks=None)
        low_res, iou_pred = self.mask_decoder(
            image_embeddings=image_emb,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False)
        return low_res, iou_pred

    def decode_coarse(self, image_emb: torch.Tensor, image_size: int):
        """Prompt-free coarse mask at image resolution (paper Eq. 1)."""
        low_res, _ = self.decode(image_emb, None, None, None)
        masks = self.postprocess_masks(
            low_res, input_size=(image_size, image_size),
            original_size=(image_size, image_size))
        return masks, low_res

    def predict_error(self, image_emb: torch.Tensor,
                      bin_mask: torch.Tensor):
        """Error map from concat[E(x), Pd(mask)] (paper Eq. 4)."""
        dense = self.mask_embedding(bin_mask)
        return self.error_decoder(torch.cat([image_emb, dense], dim=1))

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


def _load_from(sam, state_dict, image_size, vit_patch_size,
               encoder_global_attn_indexes):
    ega = encoder_global_attn_indexes
    sam_dict = sam.state_dict()
    except_keys = ['mask_tokens', 'output_hypernetworks_mlps',
                   'iou_prediction_head', 'error_decoder']
    new_state_dict = {k: v for k, v in state_dict.items()
                      if k in sam_dict.keys()
                      and all(ek not in k for ek in except_keys)}
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


def build_cosam(checkpoint=None, model_type='vit_b', image_size=256,
                num_classes=1):
    """Build CoSAM: frozen encoder + frozen prompt encoder, learnable mask
    decoder (Alg. 1: learnable M and E; frozen E, Ps, Pd)."""
    assert model_type in VIT_CONFIGS
    cfg = VIT_CONFIGS[model_type]
    prompt_embed_dim = 256
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    model = CoSAM(
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
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2, embedding_dim=prompt_embed_dim, mlp_dim=2048,
                num_heads=8),
            transformer_dim=prompt_embed_dim),
        error_decoder=ErrorDecoder(in_chans=prompt_embed_dim * 2),
        pixel_mean=[0.0, 0.0, 0.0], pixel_std=[1.0, 1.0, 1.0],
    )
    model.train()
    if checkpoint is not None:
        with open(checkpoint, 'rb') as f:
            state_dict = torch.load(f, map_location='cpu')
        _load_from(model, state_dict, image_size, vit_patch_size,
                   cfg['encoder_global_attn_indexes'])
    # Alg. 1: frozen image encoder E and prompt encoders Ps/Pd.
    for param in model.image_encoder.parameters():
        param.requires_grad = False
    for param in model.prompt_encoder.parameters():
        param.requires_grad = False
    return model
