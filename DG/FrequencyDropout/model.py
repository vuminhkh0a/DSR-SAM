"""
Frequency Dropout (FD) Model

Wraps VGG16BN U-Net with Frequency Dropout regularization layers
after each encoder block. Supports both FD-RF (randomized filtering
with Gaussian/LoG/Gabor) and FD-GF (Gaussian only for fair CBS comparison).

During eval, FD filters are fixed (from checkpoint) matching official code:
  "Frequency Dropout: Feature-Level Regularization via
   Randomized Filtering" (ECCV 2022 MCV Workshop)
"""
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from utils.models import ConvBlock
from DG.FrequencyDropout.filters import FILTER_REGISTRY, GaussianFilter


class VGG16BN_Unet_FD(nn.Module):
    """VGG16-BN U-Net with Frequency Dropout regularization.

    Architecture: VGG16BN encoder -> FD filters -> U-Net decoder
    FD filters are applied after each encoder block.
    Filters are regenerated per training step via get_new_kernels().
    """
    def __init__(self, with_vgg16bn=True, fd_cfg=None):
        super().__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        if not with_vgg16bn:
            self.down_conv1 = ConvBlock(3, 64)
            self.down_conv2 = ConvBlock(64, 128)
            self.down_conv3 = ConvBlock(128, 256)
            self.down_conv4 = ConvBlock(256, 512)
            self.down_conv5 = ConvBlock(512, 1024)
        else:
            vgg = models.vgg16_bn(weights=models.VGG16_BN_Weights.IMAGENET1K_V1)
            features = vgg.features
            self.down_conv1 = features[0:6]
            self.down_conv2 = features[7:13]
            self.down_conv3 = features[14:23]
            self.down_conv4 = features[24:33]
            self.down_conv5 = nn.Sequential(features[34:43], ConvBlock(512, 1024))

        self.ch_list = [64, 128, 256, 512, 1024]

        self.use_fd = fd_cfg is not None
        if self.use_fd:
            self.kernel_size = fd_cfg.get('kernel_size', 3)
            self.use_rf = fd_cfg.get('use_rf', True)
            self.freq_min_all = fd_cfg.get('freq_min_all', [0.2, 0.2, 0.0])
            self.freq_max_all = fd_cfg.get('freq_max_all', [1.0, 3.0, 1.0])
            self.dropout_p_all = fd_cfg.get('dropout_p_all', [0.4, 0.5, 0.8])
            self.filter_all = FILTER_REGISTRY if self.use_rf else [GaussianFilter]
            self.num_filters = len(self.filter_all) - 1
        else:
            self.num_filters = -1

        self.post_bn1 = nn.BatchNorm2d(64) if self.use_fd else nn.Identity()
        self.post_relu1 = nn.ReLU(inplace=True) if self.use_fd else nn.Identity()
        self.post_bn2 = nn.BatchNorm2d(128) if self.use_fd else nn.Identity()
        self.post_relu2 = nn.ReLU(inplace=True) if self.use_fd else nn.Identity()
        self.post_bn3 = nn.BatchNorm2d(256) if self.use_fd else nn.Identity()
        self.post_relu3 = nn.ReLU(inplace=True) if self.use_fd else nn.Identity()
        self.post_bn4 = nn.BatchNorm2d(512) if self.use_fd else nn.Identity()
        self.post_relu4 = nn.ReLU(inplace=True) if self.use_fd else nn.Identity()
        self.post_bn5 = nn.BatchNorm2d(1024) if self.use_fd else nn.Identity()
        self.post_relu5 = nn.ReLU(inplace=True) if self.use_fd else nn.Identity()

        self.fd_kernel1 = nn.Identity()
        self.fd_kernel2 = nn.Identity()
        self.fd_kernel3 = nn.Identity()
        self.fd_kernel4 = nn.Identity()
        self.fd_kernel5 = nn.Identity()

        self.up_transpose1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up_conv1 = ConvBlock(1024, 512)
        self.up_transpose2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up_conv2 = ConvBlock(512, 256)
        self.up_transpose3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_conv3 = ConvBlock(256, 128)
        self.up_transpose4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_conv4 = ConvBlock(128, 64)
        self.final = nn.Conv2d(64, 1, kernel_size=1)

        if self.use_fd:
            self._init_fd_kernels()

    def _init_fd_kernels(self):
        """Initialize FD kernels as identity (Gaussian with sigma=0.01 approx identity)."""
        init_sigma = 0.01
        self.fd_kernel1 = GaussianFilter(
            ksize=self.kernel_size, sigma=torch.tensor([init_sigma] * self.ch_list[0]),
            channels=self.ch_list[0])
        self.fd_kernel2 = GaussianFilter(
            ksize=self.kernel_size, sigma=torch.tensor([init_sigma] * self.ch_list[1]),
            channels=self.ch_list[1])
        self.fd_kernel3 = GaussianFilter(
            ksize=self.kernel_size, sigma=torch.tensor([init_sigma] * self.ch_list[2]),
            channels=self.ch_list[2])
        self.fd_kernel4 = GaussianFilter(
            ksize=self.kernel_size, sigma=torch.tensor([init_sigma] * self.ch_list[3]),
            channels=self.ch_list[3])
        self.fd_kernel5 = GaussianFilter(
            ksize=self.kernel_size, sigma=torch.tensor([init_sigma] * self.ch_list[4]),
            channels=self.ch_list[4])

    def _random_selection(self, channels, f_idx):
        dropout_p = self.dropout_p_all[f_idx]
        freq_min, freq_max = self.freq_min_all[f_idx], self.freq_max_all[f_idx]
        sigma = torch.tensor(np.random.uniform(freq_min, freq_max, channels)).float()
        if dropout_p > 0:
            mask = torch.bernoulli(torch.full((channels,), 1.0 - dropout_p))
            sigma = sigma * mask
        return sigma

    def get_new_kernels(self):
        """Randomly select filter type and generate new FD kernels for all stages.

        Called once per training iteration. Matches official FD-RF:
          - Random filter type (Gaussian / LoG / Gabor)
          - Random sigma per channel
          - Channel dropout
        """
        device = next(self.parameters()).device
        f_idx = random.randint(0, self.num_filters)
        filter_cls = self.filter_all[f_idx]
        ks = self.kernel_size

        sigma1 = self._random_selection(self.ch_list[0], f_idx)
        self.fd_kernel1 = filter_cls(ksize=ks, sigma=sigma1, channels=self.ch_list[0]).to(device)

        sigma2 = self._random_selection(self.ch_list[1], f_idx)
        self.fd_kernel2 = filter_cls(ksize=ks, sigma=sigma2, channels=self.ch_list[1]).to(device)

        sigma3 = self._random_selection(self.ch_list[2], f_idx)
        self.fd_kernel3 = filter_cls(ksize=ks, sigma=sigma3, channels=self.ch_list[2]).to(device)

        sigma4 = self._random_selection(self.ch_list[3], f_idx)
        self.fd_kernel4 = filter_cls(ksize=ks, sigma=sigma4, channels=self.ch_list[3]).to(device)

        sigma5 = self._random_selection(self.ch_list[4], f_idx)
        self.fd_kernel5 = filter_cls(ksize=ks, sigma=sigma5, channels=self.ch_list[4]).to(device)

    def forward(self, x):
        down1 = self.down_conv1(x)
        down1 = self.fd_kernel1(down1)
        down1 = self.post_relu1(self.post_bn1(down1))
        max1 = self.maxpool(down1)

        down2 = self.down_conv2(max1)
        down2 = self.fd_kernel2(down2)
        down2 = self.post_relu2(self.post_bn2(down2))
        max2 = self.maxpool(down2)

        down3 = self.down_conv3(max2)
        down3 = self.fd_kernel3(down3)
        down3 = self.post_relu3(self.post_bn3(down3))
        max3 = self.maxpool(down3)

        down4 = self.down_conv4(max3)
        down4 = self.fd_kernel4(down4)
        down4 = self.post_relu4(self.post_bn4(down4))
        max4 = self.maxpool(down4)

        down5 = self.down_conv5(max4)
        down5 = self.fd_kernel5(down5)
        down5 = self.post_relu5(self.post_bn5(down5))

        up1 = self.up_conv1(torch.cat([down4, self.up_transpose1(down5)], dim=1))
        up2 = self.up_conv2(torch.cat([down3, self.up_transpose2(up1)], dim=1))
        up3 = self.up_conv3(torch.cat([down2, self.up_transpose3(up2)], dim=1))
        up4 = self.up_conv4(torch.cat([down1, self.up_transpose4(up3)], dim=1))
        out = torch.sigmoid(self.final(up4))

        return out
