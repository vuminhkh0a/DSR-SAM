"""DSR-SAM2: dual-branch (SAM ViT + ResNet18 CNN) DSR-SAM.

The ViT branch is the unchanged SAM encoder: all ``depth`` transformer blocks
run as usual, and only the last ``cba_last_n = n`` blocks additionally host
Cross-Branch Attention (CBA). No ViT transformer block is added or removed.

The CNN branch always keeps its base: ImageNet1k-v1 ResNet18 stem (7x7 conv
+ BN + ReLU + maxpool), layer1-3 (6 pretrained ResBlocks, 16x16x256) and the
new layer4 projection (256->768, 16x16x768) -- 7 ResBlocks in total.
``cba_last_n = n`` then appends n new 768->768 ResBlocks
(layer5 ... layer(4+n), chained after layer4). The last n ViT blocks pair
one-to-one with those extra ResBlocks, and each hosts two CBA blocks
(2n total): CBA1 after norm1 with the ResBlock "before" feature, CBA2 after
norm2 with its "after" feature. Each CBA computes the
per-channel linear CKA (Centered Kernel Alignment) between the same-index
pair and an AttentionGate (Oktay et al. 2018) that uses f_vit as the gating
signal to filter f_cnn:

``f_gated_cnn = AttentionGate(f_cnn, f_vit)``
``fused = f_vit + ((1 - C_weight) * f_cnn + f_gated_cnn) / 2``

where ``C_weight`` in [0, 1] is 1 for aligned channels and 0 for
uncorrelated ones.  ``use_attention_gate=False`` falls back to
``f_vit + (1 - C_weight) * f_cnn``.

Before the SAM neck, the final ViT/CNN fusion is plain addition:
``x = x + cnn_final``, using the last extra ResBlock output as ``cnn_final``
(no CKA and no attention gate).

Training adds ``cka_loss_weight * L_CKA`` (default 0.1).  ``L_CKA`` sums
every CBA ``C_weight`` term, normalized by the literal
``batch_size * total_CBA_in_model * total_channels_in_all_CBA``.

Shared config helpers are used by every DSR-SAM2 entry point (``main``,
``main_a``, ``main_h``) and by the SOTA runner overrides in
``DG/sota/run_sota.py``.
"""
