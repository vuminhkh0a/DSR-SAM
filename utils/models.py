import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ============================================================
# SL Models (Supervised Learning - U-Net variants)
# ============================================================

class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu2(self.bn2(self.conv2(self.relu1(self.bn1(self.conv1(x))))))

class Unet(nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_conv1 = ConvBlock(3, 64)
        self.down_conv2 = ConvBlock(64, 128)
        self.down_conv3 = ConvBlock(128, 256)
        self.down_conv4 = ConvBlock(256, 512)
        self.down_conv5 = ConvBlock(512, 1024)
        self.up_transpose1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up_conv1 = ConvBlock(1024, 512)
        self.up_transpose2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up_conv2 = ConvBlock(512, 256)
        self.up_transpose3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_conv3 = ConvBlock(256, 128)
        self.up_transpose4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_conv4 = ConvBlock(128, 64)
        self.final = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        down1 = self.down_conv1(x)
        max1 = self.maxpool(down1)
        down2 = self.down_conv2(max1)
        max2 = self.maxpool(down2)
        down3 = self.down_conv3(max2)
        max3 = self.maxpool(down3)
        down4 = self.down_conv4(max3)
        max4 = self.maxpool(down4)
        down5 = self.down_conv5(max4)
        up1 = self.up_conv1(torch.cat([down4, self.up_transpose1(down5)], dim=1))
        up2 = self.up_conv2(torch.cat([down3, self.up_transpose2(up1)], dim=1))
        up3 = self.up_conv3(torch.cat([down2, self.up_transpose3(up2)], dim=1))
        up4 = self.up_conv4(torch.cat([down1, self.up_transpose4(up3)], dim=1))
        return torch.sigmoid(self.final(up4))

class VGG16BN_Unet(nn.Module):
    def __init__(self, with_vgg16bn=False):
        super().__init__()
        self.with_vgg16bn = with_vgg16bn
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        if not self.with_vgg16bn:
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
        self.up_conv1 = ConvBlock(1024, 512)
        self.up_conv2 = ConvBlock(512, 256)
        self.up_conv3 = ConvBlock(256, 128)
        self.up_conv4 = ConvBlock(128, 64)
        self.up_transpose1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up_transpose2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up_transpose3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_transpose4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.final = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        down1 = self.down_conv1(x)
        max1 = self.maxpool(down1)
        down2 = self.down_conv2(max1)
        max2 = self.maxpool(down2)
        down3 = self.down_conv3(max2)
        max3 = self.maxpool(down3)
        down4 = self.down_conv4(max3)
        max4 = self.maxpool(down4)
        down5 = self.down_conv5(max4)
        up1 = self.up_conv1(torch.cat([down4, self.up_transpose1(down5)], dim=1))
        up2 = self.up_conv2(torch.cat([down3, self.up_transpose2(up1)], dim=1))
        up3 = self.up_conv3(torch.cat([down2, self.up_transpose3(up2)], dim=1))
        up4 = self.up_conv4(torch.cat([down1, self.up_transpose4(up3)], dim=1))
        return torch.sigmoid(self.final(up4))


# ============================================================
# DG Models (Domain Generalization - Dual Normalization)
# ============================================================

class DomainSpecificBatchNorm2d(nn.Module):
    def __init__(self, num_features, num_domains, eps=1e-5, momentum=0.1, affine=True):
        super().__init__()
        self.bns = nn.ModuleList([
            nn.BatchNorm2d(num_features, eps, momentum, affine)
            for _ in range(num_domains)
        ])

    def forward(self, x, domain_label):
        bs = x.shape[0]
        out = torch.zeros_like(x)
        for d in range(len(self.bns)):
            mask = (domain_label == d)
            if mask.any():
                out[mask] = self.bns[d](x[mask])
        return out

    def get_bn_stats(self):
        means, vars = [], []
        for bn in self.bns:
            means.append(bn.running_mean.clone())
            vars.append(bn.running_var.clone())
        return means, vars


class ConvD_DN(nn.Module):
    def __init__(self, inplanes, planes, num_domains, first=False):
        super().__init__()
        self.first = first
        self.conv1 = nn.Conv2d(inplanes, planes, 3, 1, 1, bias=True)
        self.bn1   = DomainSpecificBatchNorm2d(planes, num_domains)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=True)
        self.bn2   = DomainSpecificBatchNorm2d(planes, num_domains)
        self.conv3 = nn.Conv2d(planes, planes, 3, 1, 1, bias=True)
        self.bn3   = DomainSpecificBatchNorm2d(planes, num_domains)

    def forward(self, x, domain_label):
        if not self.first:
            x = F.max_pool2d(x, 2)
        x = self.bn1(self.conv1(x), domain_label)
        y = F.relu(self.bn2(self.conv2(x), domain_label))
        z = F.relu(self.bn3(self.conv3(y), domain_label))
        return z


class ConvU_DN(nn.Module):
    def __init__(self, planes, num_domains, first=False):
        super().__init__()
        self.first = first
        if not self.first:
            self.conv1 = nn.Conv2d(2 * planes, planes, 3, 1, 1, bias=True)
            self.bn1   = DomainSpecificBatchNorm2d(planes, num_domains)
        self.conv2 = nn.Conv2d(planes, planes // 2, 1, 1, 0, bias=True)
        self.bn2   = DomainSpecificBatchNorm2d(planes // 2, num_domains)
        self.conv3 = nn.Conv2d(planes, planes, 3, 1, 1, bias=True)
        self.bn3   = DomainSpecificBatchNorm2d(planes, num_domains)

    def forward(self, x, prev, domain_label):
        if not self.first:
            x = F.relu(self.bn1(self.conv1(x), domain_label))
        y = F.interpolate(x, scale_factor=2, mode='nearest')
        y = F.relu(self.bn2(self.conv2(y), domain_label))
        y = torch.cat([prev, y], dim=1)
        y = F.relu(self.bn3(self.conv3(y), domain_label))
        return y


class Unet2D_DN(nn.Module):
    def __init__(self, in_channels=3, n=16, num_classes=1, num_domains=3, momentum=0.1):
        super().__init__()

        self.convd1 = ConvD_DN(in_channels,     n, num_domains, first=True)
        self.convd2 = ConvD_DN(n,             2*n, num_domains)
        self.convd3 = ConvD_DN(2*n,           4*n, num_domains)
        self.convd4 = ConvD_DN(4*n,           8*n, num_domains)
        self.convd5 = ConvD_DN(8*n,          16*n, num_domains)

        self.convu4 = ConvU_DN(16*n, num_domains, first=True)
        self.convu3 = ConvU_DN(8*n,  num_domains)
        self.convu2 = ConvU_DN(4*n,  num_domains)
        self.convu1 = ConvU_DN(2*n,  num_domains)

        self.seg1 = nn.Conv2d(2*n, num_classes, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x, domain_label=None):
        batch_size = x.shape[0]
        if domain_label is None:
            domain_label = torch.zeros(batch_size, dtype=torch.long, device=x.device)

        x1 = self.convd1(x, domain_label)
        x2 = self.convd2(x1, domain_label)
        x3 = self.convd3(x2, domain_label)
        x4 = self.convd4(x3, domain_label)
        x5 = self.convd5(x4, domain_label)

        y4 = self.convu4(x5, x4, domain_label)
        y3 = self.convu3(y4, x3, domain_label)
        y2 = self.convu2(y3, x2, domain_label)
        y1 = self.convu1(y2, x1, domain_label)

        out = self.seg1(y1)
        return torch.sigmoid(out)


# ============================================================
# Benchmark Encoders for IJEPA / LeJEPA segmentation
# ============================================================

ENCODER_CONFIGS = {
    'vit_tiny':    {'type': 'vit', 'embed_dim': 192,  'depth': 12, 'num_heads': 3},
    'vit_small':   {'type': 'vit', 'embed_dim': 384,  'depth': 12, 'num_heads': 6},
    'vit_base':    {'type': 'vit', 'embed_dim': 768,  'depth': 12, 'num_heads': 12},
    'vit_large':   {'type': 'vit', 'embed_dim': 1024, 'depth': 24, 'num_heads': 16},
    'vit_huge':    {'type': 'vit', 'embed_dim': 1280, 'depth': 32, 'num_heads': 16},
    'resnet18':    {'type': 'conv', 'backbone': 'resnet18',   'out_channels': 512},
    'resnet34':    {'type': 'conv', 'backbone': 'resnet34',   'out_channels': 512},
    'resnet50':    {'type': 'conv', 'backbone': 'resnet50',   'out_channels': 2048},
    'resnet101':   {'type': 'conv', 'backbone': 'resnet101',  'out_channels': 2048},
    'resnet152':   {'type': 'conv', 'backbone': 'resnet152',  'out_channels': 2048},
    'efficientnet_b0': {'type': 'conv', 'backbone': 'efficientnet_b0', 'out_channels': 1280},
    'efficientnet_b1': {'type': 'conv', 'backbone': 'efficientnet_b1', 'out_channels': 1280},
    'efficientnet_b2': {'type': 'conv', 'backbone': 'efficientnet_b2', 'out_channels': 1408},
    'efficientnet_b3': {'type': 'conv', 'backbone': 'efficientnet_b3', 'out_channels': 1536},
    'convnext_tiny':   {'type': 'conv', 'backbone': 'convnext_tiny',   'out_channels': 768},
    'convnext_small':  {'type': 'conv', 'backbone': 'convnext_small',  'out_channels': 768},
    'convnext_base':   {'type': 'conv', 'backbone': 'convnext_base',   'out_channels': 1024},
    'convnext_large':  {'type': 'conv', 'backbone': 'convnext_large',  'out_channels': 1536},
}


def get_encoder_name(cfg):
    if cfg['encoder_type'] == 'vit':
        return cfg['vit_name']
    return cfg['encoder_type']


class ConvDecoder(nn.Module):
    def __init__(self, in_channels, out_ch=1):
        super().__init__()
        self.up1 = nn.ConvTranspose2d(in_channels, 256, kernel_size=2, stride=2)
        self.conv1 = ConvBlock(256, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv2 = ConvBlock(128, 128)
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv3 = ConvBlock(64, 64)
        self.up4 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv4 = ConvBlock(32, 32)
        self.final = nn.Conv2d(32, out_ch, kernel_size=1)

    def forward(self, x):
        x = self.up1(x)
        x = self.conv1(x)
        x = self.up2(x)
        x = self.conv2(x)
        x = self.up3(x)
        x = self.conv3(x)
        x = self.up4(x)
        x = self.conv4(x)
        return torch.sigmoid(self.final(x))


class BackboneEncoder(nn.Module):
    def __init__(self, backbone_name, out_channels, pretrained=True):
        super().__init__()
        weights = 'DEFAULT' if pretrained else None
        if backbone_name.startswith('resnet'):
            model_cls = getattr(models, backbone_name)
            model = model_cls(weights=weights)
            self.stem = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool)
            self.layer1 = model.layer1
            self.layer2 = model.layer2
            self.layer3 = model.layer3
            self.layer4 = model.layer4
            self.out_channels = out_channels
        elif backbone_name.startswith('efficientnet'):
            model_cls = getattr(models, backbone_name)
            model = model_cls(weights=weights)
            self.stem = model.features[:2]
            self.layer1 = model.features[2:4]
            self.layer2 = model.features[4:6]
            self.layer3 = model.features[6:8]
            self.layer4 = model.features[8:]
            self.out_channels = out_channels
        elif backbone_name.startswith('convnext'):
            model_cls = getattr(models, backbone_name)
            model = model_cls(weights=weights)
            self.stem = model.features[:2]
            self.layer1 = model.features[2:4]
            self.layer2 = model.features[4:6]
            self.layer3 = model.features[6:8]
            self.layer4 = model.features[8:]
            self.out_channels = out_channels
        else:
            raise ValueError(f'Unsupported backbone: {backbone_name}')

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class BackboneSeg(nn.Module):
    def __init__(self, backbone_name, out_channels, decoder_in_channels=None, pretrained=True, freeze_encoder=True):
        super().__init__()
        self.encoder = BackboneEncoder(backbone_name, out_channels, pretrained=pretrained)
        dec_in = decoder_in_channels or out_channels
        self.decoder = ConvDecoder(dec_in)
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def forward(self, x):
        features = self.encoder(x)
        return self.decoder(features)

    def train(self, mode=True):
        super().train(mode)
        if mode:
            for p in self.encoder.parameters():
                p.requires_grad = False
