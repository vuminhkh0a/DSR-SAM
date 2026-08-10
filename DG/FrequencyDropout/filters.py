"""
2D Frequency Dropout Filters

Implementation of Gaussian, Laplacian of Gaussian (LoG),
and Gabor filters for Frequency Dropout (FD).

Based on:
  Paper:  "Frequency Dropout: Feature-Level Regularization via
           Randomized Filtering" (ECCV 2022 MCV Workshop)
  Repo:   https://github.com/mobarakol/Frequency_Dropout
  Baselin: https://github.com/pairlab/CBS
"""
import math
import numpy as np
import torch
import torch.nn as nn


# ============================================================================
# Gaussian Filter (2D)
# ============================================================================

def get_gaussian_kernel_2d(ksize, sigma, channels):
    x_grid = torch.arange(ksize).repeat(ksize).view(ksize, ksize)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()
    mean = (ksize - 1) / 2.0
    variance = sigma ** 2
    gk = (1.0 / (2.0 * math.pi * variance.view(channels, 1, 1) + 1e-16)) * torch.exp(
        -torch.sum((xy_grid - mean) ** 2, dim=-1)
        .view(1, ksize, ksize)
        .repeat(channels, 1, 1)
        / (2.0 * variance.view(channels, 1, 1) + 1e-16)
    )
    gk = gk / torch.sum(gk, dim=(1, 2)).view(channels, 1, 1)
    return gk.unsqueeze(1).float()


class GaussianFilter(nn.Module):
    def __init__(self, ksize, sigma, channels):
        super().__init__()
        sigma_ts = torch.tensor(sigma).repeat(channels) if np.isscalar(sigma) else sigma
        kernel = get_gaussian_kernel_2d(ksize, sigma_ts, channels)
        pad = ksize // 2
        self.conv = nn.Conv2d(channels, channels, ksize, groups=channels, bias=False, padding=pad)
        self.conv.weight.data = kernel
        self.conv.weight.requires_grad = False

    def forward(self, x):
        return self.conv(x)


# ============================================================================
# Laplacian of Gaussian (LoG) Filter (2D)
# ============================================================================

def get_log_kernel_2d(ksize, sigma, channels):
    x_grid = torch.arange(ksize).repeat(ksize).view(ksize, ksize)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()
    mean = (ksize - 1) / 2.0
    variance = sigma ** 2
    log_k = (
        -1.0
        / (math.pi * (variance ** 2).view(channels, 1, 1) + 1e-16)
        * (
            1.0
            - torch.sum((xy_grid - mean) ** 2, dim=-1)
            / (2.0 * variance.view(channels, 1, 1) + 1e-16)
        )
        * torch.exp(
            -torch.sum((xy_grid - mean) ** 2, dim=-1)
            / (2.0 * variance.view(channels, 1, 1) + 1e-16)
        )
    )
    log_k = log_k / torch.sum(log_k, dim=(1, 2)).view(channels, 1, 1)
    return log_k.unsqueeze(1).float()


class LaplacianOfGaussianFilter(nn.Module):
    def __init__(self, ksize, sigma, channels):
        super().__init__()
        kernel = get_log_kernel_2d(ksize, sigma, channels)
        pad = ksize // 2
        self.conv = nn.Conv2d(channels, channels, ksize, groups=channels, bias=False, padding=pad)
        self.conv.weight.data = kernel
        self.conv.weight.requires_grad = False

    def forward(self, x):
        return self.conv(x)


# ============================================================================
# Gabor Filter (2D)
# ============================================================================

def _gabor_2d(ksize, sigma, theta, lambd, gamma, psi, channels):
    xmax = ymax = ksize // 2
    y, x = torch.meshgrid(
        torch.arange(-xmax, xmax + 1), torch.arange(-ymax, ymax + 1), indexing='ij'
    )
    x_theta = x * torch.cos(theta) + y * torch.sin(theta)
    y_theta = -x * torch.sin(theta) + y * torch.cos(theta)
    gauss = torch.exp(
        -(x_theta ** 2 + gamma ** 2 * y_theta ** 2) / (2.0 * (sigma ** 2).view(channels, 1, 1))
    )
    grating = torch.cos(2.0 * math.pi / lambd * x_theta + psi)
    return gauss * grating


def get_gabor_kernel_2d(ksize, sigma, channels, theta=0.0, lambd=3.0, gamma=0.0):
    gk = _gabor_2d(
        ksize, sigma, torch.tensor(theta).float(), lambd, gamma, psi=0.0, channels=channels
    )
    gk = gk.unsqueeze(1).float()
    dummy = torch.zeros(ksize, ksize)
    dummy[ksize // 2, ksize // 2] = 1.0
    gk[sigma == 0] = dummy
    return gk


class GaborFilter(nn.Module):
    def __init__(self, ksize, sigma, channels, theta=0.0):
        super().__init__()
        kernel = get_gabor_kernel_2d(ksize, sigma, channels, theta=theta)
        pad = ksize // 2
        self.conv = nn.Conv2d(channels, channels, ksize, groups=channels, bias=False, padding=pad)
        self.conv.weight.data = kernel
        self.conv.weight.requires_grad = False

    def forward(self, x):
        return self.conv(x)


# ============================================================================
# Filter registry for random selection
# ============================================================================

FILTER_REGISTRY = [GaussianFilter, LaplacianOfGaussianFilter, GaborFilter]
