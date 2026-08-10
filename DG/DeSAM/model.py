"""
DeSAM: Decoupled Segment Anything Model for Generalizable Medical Image
Segmentation (MICCAI 2024, arXiv:2306.00499).

Pure-PyTorch port of the official repo (https://github.com/yifangao112/DeSAM):
SAM ViT-H image encoder + frozen prompt encoder + decoupled mask decoder.
The decoder consists of
  * PRIM (prompt-relevant IoU module): SAM's cross-attention transformer
    without the mask prediction head; it predicts the IoU score and outputs
    the mask embedding from the cross-attention transformer layer.
  * PDMM (prompt-decoupled mask module): multi-scale (SE residual blocks +
    upsampling, UNETR-style) fusion of the intermediate image embeddings
    (global attention layers 8/16/24 of ViT-H) with the PRIM mask embedding.
"""
import math
from functools import partial
from typing import Any, List, Optional, Tuple, Type

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Common building blocks (from SAM / repo desam/modeling/common.py)
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
    def __init__(self, embedding_dim: int, mlp_dim: int, act: Type[nn.Module] = nn.GELU) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def _conv2d(in_channels, out_channels, kernel_size, stride=1, is_transposed=False):
    if is_transposed:
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size,
                                  stride=stride, padding=0, bias=True)
    padding = 1 if kernel_size > 1 else 0
    return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                     stride=stride, padding=padding, bias=True)


def _instance_norm2d(channels):
    return nn.InstanceNorm2d(channels, eps=1e-5, momentum=0.1, affine=True,
                             track_running_stats=False)


# ============================================================
# PDMM: squeeze-and-excitation residual blocks (UNETR/DynUNet style,
# repo desam/modeling/mask_decoder.py, monai UnetResSEBlock)
# ============================================================

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class UnetResSEBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.conv1 = _conv2d(in_channels, out_channels, kernel_size, stride=stride)
        self.conv2 = _conv2d(out_channels, out_channels, kernel_size, stride=1)
        self.conv3 = _conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        self.lrelu = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.norm1 = _instance_norm2d(out_channels)
        self.norm2 = _instance_norm2d(out_channels)
        self.norm3 = _instance_norm2d(out_channels)
        self.downsample = in_channels != out_channels
        if not np.all(np.atleast_1d(stride) == 1):
            self.downsample = True
        self.se = SELayer(channel=out_channels)

    def forward(self, inp):
        residual = inp
        out = self.conv1(inp)
        out = self.norm1(out)
        out = self.lrelu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.se(out)
        if self.downsample:
            residual = self.conv3(residual)
            residual = self.norm3(residual)
        out += residual
        out = self.lrelu(out)
        return out


class PreUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, upsample_kernel_size=2, num_layer=1):
        super().__init__()
        self.input_channels = in_channels
        self.output_channels = out_channels
        self.block_init = UnetResSEBlock(in_channels, out_channels, kernel_size=3, stride=1)
        self.residual_block = nn.ModuleList(
            [
                nn.Sequential(
                    _conv2d(out_channels, out_channels, upsample_kernel_size,
                            stride=upsample_kernel_size, is_transposed=True),
                    UnetResSEBlock(out_channels, out_channels, kernel_size=3, stride=1),
                )
                for _ in range(num_layer)
            ]
        )

    def forward(self, x):
        x = self.block_init(x)
        for blk in self.residual_block:
            x = blk(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, upsample_kernel_size=2):
        super().__init__()
        self.input_channels = in_channels
        self.output_channels = out_channels
        self.transp_conv = _conv2d(in_channels, out_channels, upsample_kernel_size,
                                   stride=upsample_kernel_size, is_transposed=True)
        self.res_block = UnetResSEBlock(out_channels + out_channels, out_channels,
                                        kernel_size=3, stride=1)

    def forward(self, inp, skip):
        inp = self.transp_conv(inp)
        out = torch.cat((inp, skip), dim=1)
        out = self.res_block(out)
        return out


class PIMMDecoder(nn.Module):
    def __init__(self, endoder_transformer_dim=1280, upsample_transformer_dim=256,
                 sam_features_length=3, do_deep_supervision=False):
        super().__init__()
        self.sam_features_length = sam_features_length
        self.do_deep_supervision = do_deep_supervision

        self.conv_blocks_context = []
        self.conv_blocks_localization = []
        self.seg_outputs = []

        self.encoder_embed_size = [int(upsample_transformer_dim // 2 ** i)
                                   for i in range(sam_features_length)]

        for d in range(self.sam_features_length):
            in_channels = endoder_transformer_dim
            out_channels = self.encoder_embed_size[d]
            self.conv_blocks_context.append(
                PreUpBlock(in_channels=in_channels, out_channels=out_channels,
                           upsample_kernel_size=2, num_layer=d)
            )

        for d in range(self.sam_features_length - 1):
            in_channels = self.encoder_embed_size[d]
            out_channels = self.encoder_embed_size[d + 1]
            self.conv_blocks_localization.append(
                UpBlock(in_channels=in_channels, out_channels=out_channels,
                        upsample_kernel_size=2)
            )

        for ds in range(len(self.conv_blocks_localization)):
            self.seg_outputs.append(
                nn.Conv2d(self.conv_blocks_localization[ds].output_channels, 1, 1, 1, 0, bias=False)
            )

        self.mask_embedding_fusion = UnetResSEBlock(
            in_channels=self.encoder_embed_size[0] + self.encoder_embed_size[0],
            out_channels=self.encoder_embed_size[0],
            kernel_size=3,
            stride=1,
        )

        self.conv_blocks_context = nn.ModuleList(self.conv_blocks_context)
        self.conv_blocks_localization = nn.ModuleList(self.conv_blocks_localization)
        self.seg_outputs = nn.ModuleList(self.seg_outputs)

    def forward(self, mask_embeddings, image_embeddings):
        skips = []
        seg_outputs = []

        for d in range(len(self.conv_blocks_context)):
            embed = self.conv_blocks_context[d](image_embeddings[-(d + 1)])
            if d == 0:
                embed = torch.cat((mask_embeddings, embed), dim=1)
                embed = self.mask_embedding_fusion(embed)
            skips.append(embed)

        for u in range(len(self.conv_blocks_localization)):
            if u == 0:
                enc_x = skips[0]
                dec_x = skips[1]
            else:
                dec_x = skips[u + 1]
            enc_x = self.conv_blocks_localization[u](enc_x, dec_x)
            seg_outputs.append(self.seg_outputs[u](enc_x))

        if self.do_deep_supervision:
            return seg_outputs[::-1]
        return seg_outputs[-1]


# ============================================================
# Prompt encoder (from SAM / repo desam/modeling/prompt_encoder.py)
# ============================================================

class PositionEmbeddingRandom(nn.Module):
    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

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

    def forward_with_coords(self, coords_input: torch.Tensor, image_size: Tuple[int, int]) -> torch.Tensor:
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords.to(torch.float))


class PromptEncoder(nn.Module):
    def __init__(self, embed_dim, image_embedding_size, input_image_size, mask_in_chans,
                 activation: Type[nn.Module] = nn.GELU):
        super().__init__()
        self.embed_dim = embed_dim
        self.input_image_size = input_image_size
        self.image_embedding_size = image_embedding_size
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)

        self.num_point_embeddings = 4
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

    def _embed_points(self, points, labels, pad):
        points = points + 0.5
        if pad:
            padding_point = torch.zeros((points.shape[0], 1, 2), device=points.device)
            padding_label = -torch.ones((labels.shape[0], 1), device=labels.device)
            points = torch.cat([points, padding_point], dim=1)
            labels = torch.cat([labels, padding_label], dim=1)
        point_embedding = self.pe_layer.forward_with_coords(points, self.input_image_size)
        point_embedding[labels == -1] = 0.0
        point_embedding[labels == -1] += self.not_a_point_embed.weight
        point_embedding[labels == 0] += self.point_embeddings[0].weight
        point_embedding[labels == 1] += self.point_embeddings[1].weight
        return point_embedding

    def _embed_boxes(self, boxes):
        boxes = boxes + 0.5
        coords = boxes.reshape(-1, 2, 2)
        corner_embedding = self.pe_layer.forward_with_coords(coords, self.input_image_size)
        corner_embedding[:, 0, :] += self.point_embeddings[2].weight
        corner_embedding[:, 1, :] += self.point_embeddings[3].weight
        return corner_embedding

    def _embed_masks(self, masks):
        return self.mask_downscaling(masks)

    def _get_batch_size(self, points, boxes, masks):
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
        sparse_embeddings = torch.empty((bs, 0, self.embed_dim), device=self._get_device())
        if points is not None:
            coords, labels = points
            point_embeddings = self._embed_points(coords, labels, pad=(boxes is None))
            sparse_embeddings = torch.cat([sparse_embeddings, point_embeddings], dim=1)
        if boxes is not None:
            box_embeddings = self._embed_boxes(boxes)
            sparse_embeddings = torch.cat([sparse_embeddings, box_embeddings], dim=1)
        if masks is not None:
            dense_embeddings = self._embed_masks(masks)
        else:
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                bs, -1, self.image_embedding_size[0], self.image_embedding_size[1]
            )
        return sparse_embeddings, dense_embeddings


# ============================================================
# Two-way transformer (from SAM / repo desam/modeling/transformer.py)
# ============================================================

class Attention(nn.Module):
    def __init__(self, embedding_dim, num_heads, downsample_rate=1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    def _separate_heads(self, x, num_heads):
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)

    def _recombine_heads(self, x):
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)

    def forward(self, q, k, v):
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
    def __init__(self, embedding_dim, num_heads, mlp_dim=2048, activation=nn.ReLU,
                 attention_downsample_rate=2, skip_first_layer_pe=False):
        super().__init__()
        self.self_attn = Attention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.cross_attn_token_to_image = Attention(embedding_dim, num_heads,
                                                   downsample_rate=attention_downsample_rate)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.mlp = MLPBlock(embedding_dim, mlp_dim, activation)
        self.norm3 = nn.LayerNorm(embedding_dim)
        self.norm4 = nn.LayerNorm(embedding_dim)
        self.cross_attn_image_to_token = Attention(embedding_dim, num_heads,
                                                   downsample_rate=attention_downsample_rate)
        self.skip_first_layer_pe = skip_first_layer_pe

    def forward(self, queries, keys, query_pe, key_pe):
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
    def __init__(self, depth, embedding_dim, num_heads, mlp_dim, activation=nn.ReLU,
                 attention_downsample_rate=2):
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.layers = nn.ModuleList()
        for i in range(depth):
            self.layers.append(
                TwoWayAttentionBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    attention_downsample_rate=attention_downsample_rate,
                    skip_first_layer_pe=(i == 0),
                )
            )
        self.final_attn_token_to_image = Attention(embedding_dim, num_heads,
                                                   downsample_rate=attention_downsample_rate)
        self.norm_final_attn = nn.LayerNorm(embedding_dim)

    def forward(self, image_embedding, image_pe, point_embedding):
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
# PRIM + PDMM decoupled mask decoder
# (repo desam/modeling/mask_decoder.py, MaskDecoder)
# ============================================================

class MaskDecoder(nn.Module):
    def __init__(self, *, transformer_dim, transformer, num_multimask_outputs=3,
                 activation=nn.GELU, iou_head_depth=3, iou_head_hidden_dim=256,
                 endoder_transformer_dim=1280, upsample_transformer_dim=256,
                 sam_features_length=3, do_deep_supervision=False):
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer
        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )
        self.output_hypernetworks_mlps = nn.ModuleList(
            [MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
             for _ in range(self.num_mask_tokens)]
        )
        self.iou_prediction_head = MLP(transformer_dim, iou_head_hidden_dim,
                                       self.num_mask_tokens, iou_head_depth)

        self.pimm = PIMMDecoder(
            endoder_transformer_dim=endoder_transformer_dim,
            upsample_transformer_dim=upsample_transformer_dim,
            sam_features_length=sam_features_length,
            do_deep_supervision=do_deep_supervision,
        )

    def forward(self, image_embeddings, image_pe, sparse_prompt_embeddings,
                dense_prompt_embeddings, multimask_output):
        mask_embedding, iou_pred = self.predict_masks(
            image_embeddings=image_embeddings[-1],
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
        )
        masks = self.pimm(mask_embedding, image_embeddings[:-1])
        if multimask_output:
            raise ValueError('DeSAM does not support multimask output.')
        iou_pred = iou_pred[:, slice(0, 1)]
        return masks, iou_pred

    def predict_masks(self, image_embeddings, image_pe, sparse_prompt_embeddings,
                      dense_prompt_embeddings):
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        if image_embeddings.shape[0] != tokens.shape[0]:
            src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        else:
            src = image_embeddings

        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        hs, src = self.transformer(src, pos_src, tokens)
        src = src.transpose(1, 2).view(b, c, h, w)

        iou_token_out = hs[:, 0, :]
        iou_pred = self.iou_prediction_head(iou_token_out)
        return src, iou_pred


# ============================================================
# ViT-H image encoder (from SAM / repo desam/modeling/image_encoder.py)
# ============================================================

class ViTAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, use_rel_pos=False,
                 rel_pos_zero_init=True, input_size=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert input_size is not None
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

    def forward(self, x):
        B, H, W, _ = x.shape
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)
        attn = (q * self.scale) @ k.transpose(-2, -1)
        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))
        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = self.proj(x)
        return x


def window_partition(x, window_size):
    B, H, W, C = x.shape
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w
    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows, (Hp, Wp)


def window_unpartition(windows, window_size, pad_hw, hw):
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)
    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x


def get_rel_pos(q_size, k_size, rel_pos):
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


def add_decomposed_rel_pos(attn, q, rel_pos_h, rel_pos_w, q_size, k_size):
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)
    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)
    attn = (attn.view(B, q_h, q_w, k_h, k_w) + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]
            ).view(B, q_h * q_w, k_h * k_w)
    return attn


class PatchEmbed(nn.Module):
    def __init__(self, kernel_size=(16, 16), stride=(16, 16), padding=(0, 0),
                 in_chans=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size,
                              stride=stride, padding=padding)

    def forward(self, x):
        x = self.proj(x)
        return x.permute(0, 2, 3, 1)


class ViTBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=True,
                 norm_layer=nn.LayerNorm, act_layer=nn.GELU, use_rel_pos=False,
                 rel_pos_zero_init=True, window_size=0, input_size=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = ViTAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )
        self.norm2 = norm_layer(dim)
        self.mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer)
        self.window_size = window_size

    def forward(self, x):
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
    def __init__(self, img_size=1024, patch_size=16, in_chans=3, embed_dim=768,
                 depth=12, num_heads=12, mlp_ratio=4.0, out_chans=256, qkv_bias=True,
                 norm_layer=nn.LayerNorm, act_layer=nn.GELU, use_abs_pos=True,
                 use_rel_pos=False, rel_pos_zero_init=True, window_size=0,
                 global_attn_indexes=()):
        super().__init__()
        self.img_size = img_size
        self.global_attn_indexes = global_attn_indexes

        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size), stride=(patch_size, patch_size),
            in_chans=in_chans, embed_dim=embed_dim,
        )

        self.pos_embed = None
        if use_abs_pos:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, img_size // patch_size, img_size // patch_size, embed_dim)
            )

        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = ViTBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, norm_layer=norm_layer, act_layer=act_layer,
                use_rel_pos=use_rel_pos, rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i not in global_attn_indexes else 0,
                input_size=(img_size // patch_size, img_size // patch_size),
            )
            self.blocks.append(block)

        self.neck = nn.Sequential(
            nn.Conv2d(embed_dim, out_chans, kernel_size=1, bias=False),
            LayerNorm2d(out_chans),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(out_chans),
        )

    def forward(self, x):
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed

        output_embeddings = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in self.global_attn_indexes[:-1]:
                output_embeddings.append(x.permute(0, 3, 1, 2))

        x = self.neck(x.permute(0, 3, 1, 2))
        output_embeddings.append(x)
        return output_embeddings


# ============================================================
# DeSAM model (SAM ViT-H + decoupled decoder)
# ============================================================

VIT_H_CONFIG = dict(
    encoder_embed_dim=1280, encoder_depth=32, encoder_num_heads=16,
    encoder_global_attn_indexes=[7, 15, 23, 31],
)


def build_desam(checkpoint=None, model_type='vit_h'):
    cfg = dict(VIT_H_CONFIG)
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    prompt_embed_dim = 256

    model = DeSAM(
        image_encoder=ImageEncoderViT(
            depth=cfg['encoder_depth'], embed_dim=cfg['encoder_embed_dim'],
            img_size=image_size, mlp_ratio=4,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            num_heads=cfg['encoder_num_heads'], patch_size=vit_patch_size,
            qkv_bias=True, use_rel_pos=True,
            global_attn_indexes=cfg['encoder_global_attn_indexes'],
            window_size=14, out_chans=prompt_embed_dim,
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2, embedding_dim=prompt_embed_dim,
                mlp_dim=2048, num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3, iou_head_hidden_dim=256,
            endoder_transformer_dim=cfg['encoder_embed_dim'],
            upsample_transformer_dim=256, sam_features_length=3,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )
    model.eval()

    if checkpoint is not None:
        with open(checkpoint, 'rb') as f:
            state_dict = torch.load(f, map_location='cpu')
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items()
                           if k in model_dict and 'pimm' not in k}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    return model


class DeSAM(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(self, image_encoder, prompt_encoder, mask_decoder,
                 pixel_mean=None, pixel_std=None):
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        if pixel_mean is None:
            pixel_mean = [123.675, 116.28, 103.53]
        if pixel_std is None:
            pixel_std = [58.395, 57.12, 57.375]
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

    @property
    def device(self) -> Any:
        return self.pixel_mean.device

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.pixel_mean) / self.pixel_std
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    @torch.no_grad()
    def encode_image(self, images_256: torch.Tensor, fp16: bool = True) -> List[torch.Tensor]:
        """Encode a (B, 3, 256, 256) float [0,1] batch with the frozen ViT-H
        encoder. Returns [intermediate global-attn features (x3), neck].
        The encoder runs in fp16 (autocast) to fit GPU memory; outputs are
        returned as fp32 (same values)."""
        x = images_256 * 255.0
        x = F.interpolate(x, (self.image_encoder.img_size, self.image_encoder.img_size),
                          mode='bilinear', align_corners=False)
        x = self.preprocess(x)
        if fp16 and x.is_cuda:
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                return [f.float() for f in self.image_encoder(x)]
        return self.image_encoder(x)

    def get_dense_pe(self) -> torch.Tensor:
        return self.prompt_encoder.get_dense_pe()

    def embed_points(self, coords: torch.Tensor, labels: torch.Tensor):
        return self.prompt_encoder(points=(coords, labels), boxes=None, masks=None)

    def embed_boxes(self, boxes: torch.Tensor):
        return self.prompt_encoder(points=None, boxes=boxes, masks=None)

    def decode(self, image_embeddings, sparse_embeddings, dense_embeddings):
        return self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
