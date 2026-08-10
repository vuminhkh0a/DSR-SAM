import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from utils.models import ConvBlock


class EncoderDC(nn.Module):
    """Domain code prediction branch.

    Maps encoder features -> K-dim domain similarity vector.
    Architecture matches GitHub: AdaptiveMaxPool -> BN -> ReLU -> Conv1x1(K).
    """
    def __init__(self, in_channels, num_domains):
        super().__init__()
        self.pool = nn.AdaptiveMaxPool2d((1, 1))
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()
        self.cls = nn.Conv2d(in_channels, num_domains, kernel_size=1)

    def forward(self, x):
        x = self.pool(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.cls(x)
        return x.squeeze(-1).squeeze(-1)


class MetaEmbedding(nn.Module):
    """Domain-oriented feature embedding module.

    Uses domain code to attend over centroids (hallucinator),
    then selectively infuses the retrieved feature (selector).
    Matches the exact logic from networks/MetaEmbedding.py.
    """
    def __init__(self, feat_dim, num_domains):
        super().__init__()
        self.selector = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim, kernel_size=1, bias=False),
            nn.Tanh(),
        )

    def forward(self, x, domain_code, centroids):
        direct_feature = x
        hal_scale = torch.softmax(domain_code, dim=-1)

        B, C, H, W = x.shape
        K = centroids.size(0)
        centroids_flat = centroids.view(K, -1)
        memory_feature = torch.matmul(hal_scale, centroids_flat)
        memory_feature = memory_feature.view(B, C, H, W)

        sel_scale = self.selector(x)
        infused_feature = memory_feature * sel_scale
        x = direct_feature + infused_feature
        return x, hal_scale, sel_scale


class VGG16BN_DoFE(nn.Module):
    """VGG16-BN U-Net with DoFE (Domain-oriented Feature Embedding).

    Architecture:
      VGG16BN encoder -> 1x1 proj (1024->feat_dim) ->
        EncoderDC (domain code) + MetaEmbedding (centroid retrieval) ->
        1x1 proj back (feat_dim->1024) -> U-Net decoder

    The centroids store per-domain prototype features in the projected space,
    updated via momentum during training.
    """
    def __init__(self, num_domains=2, with_vgg16bn=True, feat_dim=256, feat_hw=None):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_domains = num_domains
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

        self.proj_in = nn.Conv2d(1024, feat_dim, kernel_size=1)
        self.bn_proj = nn.BatchNorm2d(feat_dim)
        self.relu_proj = nn.ReLU()

        self.encoder_dc = EncoderDC(feat_dim, num_domains)
        self.meta_emb = MetaEmbedding(feat_dim, num_domains)

        c_hw = feat_hw if feat_hw is not None else 16
        self.centroids = nn.Parameter(
            torch.randn(num_domains, feat_dim, c_hw, c_hw), requires_grad=False
        )
        self.centroids_initialized = False

        self.proj_out = nn.Conv2d(feat_dim, 1024, kernel_size=1)

        self.up_transpose1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up_conv1 = ConvBlock(1024, 512)
        self.up_transpose2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up_conv2 = ConvBlock(512, 256)
        self.up_transpose3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_conv3 = ConvBlock(256, 128)
        self.up_transpose4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_conv4 = ConvBlock(128, 64)
        self.final = nn.Conv2d(64, 1, kernel_size=1)

    def update_memory(self, feature, domain_labels, lam=0.9):
        with torch.no_grad():
            for d in range(self.num_domains):
                mask = (domain_labels == d)
                if mask.any():
                    f = feature[mask]
                    f = f.mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
                    f_mean = f.mean(dim=0, keepdim=True)
                    self.centroids.data[d:d+1] = (
                        lam * self.centroids.data[d:d+1]
                        + (1 - lam) * f_mean
                    )

    def init_centroids_from_features(self, features_dict):
        with torch.no_grad():
            for d in range(self.num_domains):
                if d in features_dict and len(features_dict[d]) > 0:
                    all_feats = torch.stack(features_dict[d], dim=0)
                    centroid = all_feats.mean(dim=0)
                    centroid = centroid.mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
                    centroid = centroid.expand(-1, -1, self.centroids.size(2), self.centroids.size(3))
                    self.centroids.data[d:d+1] = centroid
            self.centroids_initialized = True

    def load_encoder_weights(self, state_dict):
        own_dict = self.state_dict()
        for name, param in state_dict.items():
            if name in own_dict and param.shape == own_dict[name].shape:
                own_dict[name].copy_(param)

    def forward(self, x, domain_labels=None, extract_feature=False, update_memory=False, lam=0.9):
        down1 = self.down_conv1(x)
        max1 = self.maxpool(down1)
        down2 = self.down_conv2(max1)
        max2 = self.maxpool(down2)
        down3 = self.down_conv3(max2)
        max3 = self.maxpool(down3)
        down4 = self.down_conv4(max3)
        max4 = self.maxpool(down4)
        down5 = self.down_conv5(max4)

        feat = self.relu_proj(self.bn_proj(self.proj_in(down5)))

        if extract_feature:
            return feat

        domain_code = self.encoder_dc(feat)
        fused, hal_scale, sel_scale = self.meta_emb(feat, domain_code, self.centroids)
        aug_down5 = self.proj_out(fused)

        if update_memory and domain_labels is not None and self.training:
            self.update_memory(feat.detach(), domain_labels, lam)

        up1 = self.up_conv1(torch.cat([down4, self.up_transpose1(aug_down5)], dim=1))
        up2 = self.up_conv2(torch.cat([down3, self.up_transpose2(up1)], dim=1))
        up3 = self.up_conv3(torch.cat([down2, self.up_transpose3(up2)], dim=1))
        up4 = self.up_conv4(torch.cat([down1, self.up_transpose4(up3)], dim=1))
        out = torch.sigmoid(self.final(up4))

        return out, domain_code, hal_scale, sel_scale
