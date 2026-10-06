"""
MaxStyle: Adversarial Style Composition for Robust Medical Image
Segmentation (MICCAI 2022, arXiv:2206.01737, repo cherise215/MaxStyle,
src/advanced/maxstyle.py).

The MaxStyle module (copied from the repo, device made input-dependent)
is attached to the encoder stages of a standard binary UNet: with prob p
it mixes instance styles across the batch (learnable mixing weight) and
adds learnable style noise, expanding the style space. The style
parameters are adversarially optimized to maximize the seg loss
(n_iter inner Adam steps, repo config MICCAI2022_MaxStyle.json), then the
network is trained on the stylized features.
"""
import torch
import torch.nn as nn


# ============================================================
# MaxStyle (repo src/advanced/maxstyle.py)
# ============================================================

class MaxStyle(nn.Module):
    def __init__(self, batch_size, num_feature, p=0.5, mix_style=True,
                 no_noise=False, mix_learnable=True, noise_learnable=True,
                 always_use_beta=False, alpha=0.1, eps=1e-6, debug=False):
        super().__init__()
        self.batch_size = batch_size
        self.num_feature = num_feature
        self.p = p
        self.mix_style = mix_style
        self.no_noise = no_noise
        self.mix_learnable = mix_learnable
        self.noise_learnable = noise_learnable
        self.always_use_beta = always_use_beta
        self.alpha = alpha
        self.eps = eps
        self.debug = debug
        self.data = None
        self.init_parameters()

    def _dev(self, ref):
        return ref.device

    def init_parameters(self):
        batch_size = self.batch_size
        num_feature = self.num_feature
        self.perm = torch.randperm(batch_size)
        while torch.allclose(self.perm, torch.arange(batch_size)):
            self.perm = torch.randperm(batch_size)
        self.rand_p = torch.rand(1)

        if self.rand_p >= self.p:
            self.gamma_noise = torch.zeros(batch_size, num_feature, 1, 1)
            self.beta_noise = torch.zeros(batch_size, num_feature, 1, 1)
            self.lmda = torch.zeros(batch_size, 1, 1, 1).float()
            self.gamma_noise.requires_grad = False
            self.beta_noise.requires_grad = False
            self.lmda.requires_grad = False
        else:
            if self.no_noise:
                gamma_noise = torch.randn(batch_size, num_feature, 1, 1)
                beta_noise = torch.randn(batch_size, num_feature, 1, 1)
            else:
                gamma_noise = torch.zeros(batch_size, num_feature, 1, 1)
                beta_noise = torch.zeros(batch_size, num_feature, 1, 1)
            self.gamma_noise = None
            self.beta_noise = None
            if self.noise_learnable:
                assert self.no_noise is False
                self.gamma_noise = nn.Parameter(
                    torch.empty(batch_size, num_feature, 1, 1))
                self.beta_noise = nn.Parameter(
                    torch.empty(batch_size, num_feature, 1, 1))
                nn.init.normal_(self.gamma_noise)
                nn.init.normal_(self.beta_noise)
                self.gamma_noise.requires_grad = True
                self.beta_noise.requires_grad = True
            else:
                self.gamma_noise = gamma_noise
                self.beta_noise = beta_noise
                self.gamma_noise.requires_grad = False
                self.beta_noise.requires_grad = False
            if self.mix_style is False:
                self.lmda = torch.zeros(batch_size, 1, 1, 1,
                                        dtype=torch.float32)
                self.lmda.requires_grad = False
            else:
                self.lmda = None
                if self.always_use_beta:
                    self.beta_sampler = torch.distributions.Beta(self.alpha,
                                                                 self.alpha)
                    lmda = self.beta_sampler.sample((batch_size, 1, 1, 1))
                    self.lmda = nn.Parameter(lmda.float())
                else:
                    lmda = torch.rand(batch_size, 1, 1, 1,
                                      dtype=torch.float32)
                    self.lmda = nn.Parameter(lmda.float())
                if self.mix_learnable:
                    self.lmda.requires_grad = True
                else:
                    self.lmda.requires_grad = False
        self.gamma_std = None
        self.beta_std = None

    def reset(self):
        self.init_parameters()

    def _to(self, ref):
        dev = ref.device
        self.perm = self.perm.to(dev)
        for name in ('gamma_noise', 'beta_noise', 'lmda'):
            v = getattr(self, name, None)
            if torch.is_tensor(v):
                setattr(self, name, v.to(dev))
        return dev

    def forward(self, x):
        self._to(x)
        self.data = x
        B = x.size(0)
        C = x.size(1)
        flatten_feature = x.view(B, C, -1)

        if ((self.rand_p >= self.p) or (not self.mix_style and self.no_noise)
                or B <= 1 or flatten_feature.size(2) == 1):
            return x

        assert self.batch_size == B and self.num_feature == C, \
            f'check input dim, expect ({self.batch_size}, {self.num_feature}, *,*), got {B}{C}'

        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sig = (var + self.eps).sqrt()
        mu, sig = mu.detach(), sig.detach()
        x_normed = (x - mu) / sig

        if self.gamma_std is None:
            self.gamma_std = torch.std(sig, dim=0, keepdim=True).detach()
        if self.beta_std is None:
            self.beta_std = torch.std(mu, dim=0, keepdim=True).detach()

        if B > 1:
            if self.mix_style:
                clipped_lmda = torch.clamp(self.lmda, 0, 1)
                mu2, sig2 = mu[self.perm], sig[self.perm]
                sig_mix = sig * (1 - clipped_lmda) + sig2 * clipped_lmda
                mu_mix = mu * (1 - clipped_lmda) + mu2 * clipped_lmda
            else:
                sig_mix = sig
                mu_mix = mu
            if self.no_noise:
                x_aug = sig_mix * x_normed + mu_mix
            else:
                x_aug = ((sig_mix + self.gamma_noise * self.gamma_std)
                         * x_normed
                         + (mu_mix + self.beta_noise * self.beta_std))
        else:
            x_aug = ((sig + self.gamma_noise * self.gamma_std) * x_normed
                     + (mu + self.beta_noise * self.beta_std))
        return x_aug


# ============================================================
# Binary UNet backbone carrying the MaxStyle layers
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


class MaxStyleUNet(nn.Module):
    """4-stage UNet; MaxStyle applied to encoder-stage outputs when a
    style dict {stage_idx: MaxStyle} is passed (train), else plain."""

    STAGE_CHANNELS = (64, 128, 256, 512)

    def __init__(self, in_ch=3, num_classes=1):
        super().__init__()
        C = self.STAGE_CHANNELS
        self.enc0 = _DoubleConv(in_ch, C[0])
        self.pool0 = nn.MaxPool2d(2)
        self.enc1 = _DoubleConv(C[0], C[1])
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = _DoubleConv(C[1], C[2])
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = _DoubleConv(C[2], C[3])
        self.pool3 = nn.MaxPool2d(2)
        self.bottleneck = _DoubleConv(C[3], C[3] * 2)
        self.up3 = nn.ConvTranspose2d(C[3] * 2, C[3], 2, stride=2)
        self.dec3 = _DoubleConv(C[3] * 2, C[3])
        self.up2 = nn.ConvTranspose2d(C[3], C[2], 2, stride=2)
        self.dec2 = _DoubleConv(C[2] * 2, C[2])
        self.up1 = nn.ConvTranspose2d(C[2], C[1], 2, stride=2)
        self.dec1 = _DoubleConv(C[1] * 2, C[1])
        self.up0 = nn.ConvTranspose2d(C[1], C[0], 2, stride=2)
        self.dec0 = _DoubleConv(C[0] * 2, C[0])
        self.out = nn.Conv2d(C[0], num_classes, 1)

    def _enc(self, x, style=None):
        s = style or {}
        e0 = self.enc0(x)
        if 0 in s and s[0] is not None:
            e0 = s[0](e0)
        e1 = self.enc1(self.pool0(e0))
        if 1 in s and s[1] is not None:
            e1 = s[1](e1)
        e2 = self.enc2(self.pool1(e1))
        if 2 in s and s[2] is not None:
            e2 = s[2](e2)
        e3 = self.enc3(self.pool2(e2))
        if 3 in s and s[3] is not None:
            e3 = s[3](e3)
        b = self.bottleneck(self.pool3(e3))
        return e0, e1, e2, e3, b

    def forward(self, x, style=None):
        e0, e1, e2, e3, b = self._enc(x, style)
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        d0 = self.dec0(torch.cat([self.up0(d1), e0], dim=1))
        return self.out(d0)


def build_maxstyle_unet(in_ch=3, num_classes=1):
    return MaxStyleUNet(in_ch=in_ch, num_classes=num_classes)


def make_style_modules(batch_size, device=None):
    """Fresh MaxStyle layers for the 4 encoder stages (repo config:
    p=0.5, mix + learnable noise, beta-sampled mixing weight)."""
    mods = {i: MaxStyle(batch_size, c, p=0.5, mix_style=True,
                       no_noise=False, mix_learnable=True,
                       noise_learnable=True, always_use_beta=True,
                       alpha=0.1)
            for i, c in enumerate(MaxStyleUNet.STAGE_CHANNELS)}
    if device is not None:
        for m in mods.values():
            m.to(device)
    return mods
