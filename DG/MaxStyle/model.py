import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from utils.models import ConvBlock


class MaxStyleLayer(nn.Module):
    def __init__(self, batch_size, num_features, p=0.5, mix_style=True,
                 no_noise=False, mix_learnable=True, noise_learnable=True,
                 alpha=0.1, eps=1e-6):
        super().__init__()
        self.p = p
        self.mix_style = mix_style
        self.no_noise = no_noise
        self.mix_learnable = mix_learnable
        self.noise_learnable = noise_learnable
        self.alpha = alpha
        self.eps = eps
        self.init_params(batch_size, num_features)

    def init_params(self, batch_size, num_features):
        self.rand_p = torch.rand(1).item()
        if self.rand_p >= self.p:
            self.lmda = nn.Parameter(torch.zeros(batch_size, 1, 1, 1), requires_grad=False)
            self.gamma_noise = nn.Parameter(torch.zeros(batch_size, num_features, 1, 1), requires_grad=False)
            self.beta_noise = nn.Parameter(torch.zeros(batch_size, num_features, 1, 1), requires_grad=False)
            return
        self.perm = torch.randperm(batch_size)
        while torch.allclose(self.perm.float(), torch.arange(batch_size).float()):
            self.perm = torch.randperm(batch_size)
        if self.mix_style:
            lmda = torch.rand(batch_size, 1, 1, 1)
            self.lmda = nn.Parameter(lmda, requires_grad=self.mix_learnable)
        else:
            self.lmda = nn.Parameter(torch.zeros(batch_size, 1, 1, 1), requires_grad=False)
        if self.no_noise:
            self.gamma_noise = nn.Parameter(torch.zeros(batch_size, num_features, 1, 1), requires_grad=False)
            self.beta_noise = nn.Parameter(torch.zeros(batch_size, num_features, 1, 1), requires_grad=False)
        else:
            self.gamma_noise = nn.Parameter(torch.randn(batch_size, num_features, 1, 1), requires_grad=self.noise_learnable)
            self.beta_noise = nn.Parameter(torch.randn(batch_size, num_features, 1, 1), requires_grad=self.noise_learnable)
        self.gamma_std = None
        self.beta_std = None

    def reset(self):
        if hasattr(self, 'lmda'):
            self.init_params(self.lmda.size(0), self.gamma_noise.size(1))

    def forward(self, x):
        B, C = x.size(0), x.size(1)
        if self.rand_p >= self.p or B <= 1:
            return x
        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sig = (var + self.eps).sqrt()
        mu, sig = mu.detach(), sig.detach()
        x_normed = (x - mu) / sig
        if self.gamma_std is None:
            self.gamma_std = torch.std(sig, dim=0, keepdim=True).detach()
        if self.beta_std is None:
            self.beta_std = torch.std(mu, dim=0, keepdim=True).detach()
        if B > 1 and self.mix_style:
            clipped_lmda = torch.clamp(self.lmda, 0, 1)
            mu2, sig2 = mu[self.perm], sig[self.perm]
            sig_mix = sig * (1 - clipped_lmda) + sig2 * clipped_lmda
            mu_mix = mu * (1 - clipped_lmda) + mu2 * clipped_lmda
        else:
            sig_mix, mu_mix = sig, mu
        if self.no_noise:
            x_aug = sig_mix * x_normed + mu_mix
        else:
            x_aug = (sig_mix + self.gamma_noise * self.gamma_std) * x_normed + \
                    (mu_mix + self.beta_noise * self.beta_std)
        return x_aug


class ImageDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv1 = ConvBlock(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv2 = ConvBlock(128, 64)
        self.up3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv3 = ConvBlock(32, 16)
        self.up4 = nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2)
        self.final = nn.Conv2d(8, 3, kernel_size=1)

    def forward(self, x):
        x = F.relu(self.up1(x))
        x = self.conv1(x)
        x = F.relu(self.up2(x))
        x = self.conv2(x)
        x = F.relu(self.up3(x))
        x = self.conv3(x)
        x = F.relu(self.up4(x))
        return torch.sigmoid(self.final(x))

    def forward_with_maxstyle(self, x, maxstyle_layers):
        x = F.relu(self.up1(x))
        x = self.conv1(x)
        x = F.relu(self.up2(x))
        x = self.conv2(x)
        if 'conv2' in maxstyle_layers:
            x = maxstyle_layers['conv2'](x)
        x = F.relu(self.up3(x))
        x = self.conv3(x)
        if 'conv3' in maxstyle_layers:
            x = maxstyle_layers['conv3'](x)
        x = F.relu(self.up4(x))
        if 'final' in maxstyle_layers:
            x = maxstyle_layers['final'](x)
        return torch.sigmoid(self.final(x))


class VGG16BN_Unet_MaxStyle(nn.Module):
    def __init__(self, with_vgg16bn=True):
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
        self.up_transpose1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up_conv1 = ConvBlock(1024, 512)
        self.up_transpose2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up_conv2 = ConvBlock(512, 256)
        self.up_transpose3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_conv3 = ConvBlock(256, 128)
        self.up_transpose4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_conv4 = ConvBlock(128, 64)
        self.seg_final = nn.Conv2d(64, 1, kernel_size=1)
        self.img_decoder = ImageDecoder()
        self.channel_map = {'conv2': 64, 'conv3': 16, 'final': 8}

    def encode(self, x):
        down1 = self.down_conv1(x)
        max1 = self.maxpool(down1)
        down2 = self.down_conv2(max1)
        max2 = self.maxpool(down2)
        down3 = self.down_conv3(max2)
        max3 = self.maxpool(down3)
        down4 = self.down_conv4(max3)
        max4 = self.maxpool(down4)
        down5 = self.down_conv5(max4)
        return (down1, down2, down3, down4, down5)

    def decode_seg(self, down1, down2, down3, down4, down5):
        up1 = self.up_conv1(torch.cat([down4, self.up_transpose1(down5)], dim=1))
        up2 = self.up_conv2(torch.cat([down3, self.up_transpose2(up1)], dim=1))
        up3 = self.up_conv3(torch.cat([down2, self.up_transpose3(up2)], dim=1))
        up4 = self.up_conv4(torch.cat([down1, self.up_transpose4(up3)], dim=1))
        return torch.sigmoid(self.seg_final(up4))

    def forward(self, x, maxstyle_layers=None):
        down1, down2, down3, down4, down5 = self.encode(x)
        seg = self.decode_seg(down1, down2, down3, down4, down5)
        return seg

    def forward_aug(self, x, maxstyle_layers):
        down1, down2, down3, down4, down5 = self.encode(x)
        recon = self.img_decoder.forward_with_maxstyle(down5, maxstyle_layers)
        seg_aug = self.decode_seg(down1, down2, down3, down4, down5)
        return seg_aug, recon

    def recon_forward(self, x):
        _, _, _, _, down5 = self.encode(x)
        return self.img_decoder(down5)
