"""DSR-SAM2: dual-branch (SAM ViT + ResNet18 CNN) DSR-SAM.

The ViT branch is the unchanged SAM encoder. The CNN branch is an
ImageNet1k-v1 ResNet18 up to layer3, then five new ResBlocks (layer4-8)
mapping to 16x16x768. The last four ViT blocks and ResBlocks layer5-8 are
coupled by 8 Cross-Branch Attention (CBA) blocks whose fusion is weighted by
the per-channel Grassmann geodesic distance.

Shared config helpers are used by every DSR-SAM2 entry point (``main``,
``main_a``, ``main_h``) and by the SOTA runner overrides in
``DG/sota/run_sota.py``.
"""
