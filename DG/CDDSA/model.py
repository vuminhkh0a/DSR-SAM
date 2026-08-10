import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_bn_lrelu(in_channels, out_channels, kernel_size=3, stride=1, padding=0):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.2, inplace=True)
    )


def conv_bn_relu(in_channels, out_channels, kernel_size=3, stride=1, padding=0):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


def upconv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, 1, 1),
        nn.BatchNorm2d(out_channels)
    )


def conv_block_unet(in_channels, out_channels, kernel_size, stride=1, padding=0):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(inplace=True),
    )


def conv_preactivation_relu(in_channels, out_channels, kernel_size=1, stride=1, padding=0):
    return nn.Sequential(
        nn.ReLU(inplace=False),
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.BatchNorm2d(out_channels)
    )


class Interpolate(nn.Module):
    def __init__(self, size, mode='bilinear'):
        super().__init__()
        self.size = size
        self.mode = mode

    def forward(self, x):
        return F.interpolate(x, size=self.size, mode=self.mode)


class ResConv(nn.Module):
    def __init__(self, ndf):
        super().__init__()
        self.conv1 = conv_preactivation_relu(ndf, ndf * 2, 3, 1, 1)
        self.conv2 = conv_preactivation_relu(ndf * 2, ndf * 2, 3, 1, 1)
        self.resconv = conv_preactivation_relu(ndf, ndf * 2, 1, 1, 0)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        residual = self.resconv(residual)
        return out + residual


class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.weight = None
        self.bias = None
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        assert self.weight is not None and self.bias is not None
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])
        out = F.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias,
            True, self.momentum, self.eps)
        return out.view(b, c, *x.size()[2:])


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, dim=256):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.fc3 = nn.Linear(dim, output_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        out = F.relu(self.fc1(x), inplace=True)
        out = F.relu(self.fc2(out), inplace=True)
        out = F.relu(self.fc3(out), inplace=True)
        return out


class Decoder(nn.Module):
    def __init__(self, dim, out_channel):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, 3, 1, 1, bias=True)
        self.adain1 = AdaptiveInstanceNorm2d(dim)
        self.conv2 = nn.Conv2d(dim, dim, 3, 1, 1, bias=True)
        self.adain2 = AdaptiveInstanceNorm2d(dim)
        self.conv3 = nn.Conv2d(dim, dim, 3, 1, 1, bias=True)
        self.adain3 = AdaptiveInstanceNorm2d(dim)
        self.conv4 = nn.Conv2d(dim, out_channel, 3, 1, 1, bias=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.adain1(out)
        out = self.conv2(out)
        out = self.adain2(out)
        out = self.conv3(out)
        out = self.adain3(out)
        out = self.conv4(out)
        return torch.tanh(out)


class Ada_Decoder(nn.Module):
    def __init__(self, anatomy_out_channel, z_length, out_channel):
        super().__init__()
        self.dec = Decoder(anatomy_out_channel, out_channel)
        self.mlp = MLP(z_length, self.get_num_adain_params(self.dec), 256)

    def forward(self, anatomy, style):
        adain_params = self.mlp(style)
        self.assgin_adain_params(adain_params, self.dec)
        return self.dec(anatomy)

    def get_num_adain_params(self, model):
        num = 0
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                num += 2 * m.num_features
        return num

    def assgin_adain_params(self, adain_params, model):
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                mean = adain_params[:, :m.num_features]
                std = adain_params[:, m.num_features:2 * m.num_features]
                m.bias = mean.contiguous().view(-1)
                m.weight = std.contiguous().view(-1)
                if adain_params.size(1) > 2 * m.num_features:
                    adain_params = adain_params[:, 2 * m.num_features:]


class UNet(nn.Module):
    def __init__(self, in_channel, height=256, width=256, ndf=16, num_output_channels=8):
        super().__init__()
        self.in_channel = in_channel
        self.h = height
        self.w = width
        self.ndf = ndf
        self.num_output_channels = num_output_channels

        self.encoder_block1 = conv_block_unet(self.in_channel, self.ndf, 3, 1, 1)
        self.encoder_block2 = conv_block_unet(self.ndf, self.ndf * 2, 3, 1, 1)
        self.encoder_block3 = conv_block_unet(self.ndf * 2, self.ndf * 4, 3, 1, 1)
        self.encoder_block4 = conv_block_unet(self.ndf * 4, self.ndf * 8, 3, 1, 1)
        self.maxpool = nn.MaxPool2d(2, 2)

        self.bottleneck = ResConv(self.ndf * 8)

        self.decoder_upsample1 = Interpolate((self.h // 8, self.w // 8))
        self.decoder_upconv1 = upconv(self.ndf * 16, self.ndf * 8)
        self.decoder_block1 = conv_block_unet(self.ndf * 16, self.ndf * 8, 3, 1, 1)
        self.decoder_upsample2 = Interpolate((self.h // 4, self.w // 4))
        self.decoder_upconv2 = upconv(self.ndf * 8, self.ndf * 4)
        self.decoder_block2 = conv_block_unet(self.ndf * 8, self.ndf * 4, 3, 1, 1)
        self.decoder_upsample3 = Interpolate((self.h // 2, self.w // 2))
        self.decoder_upconv3 = upconv(self.ndf * 4, self.ndf * 2)
        self.decoder_block3 = conv_block_unet(self.ndf * 4, self.ndf * 2, 3, 1, 1)
        self.decoder_upsample4 = Interpolate((self.h, self.w))
        self.decoder_upconv4 = upconv(self.ndf * 2, self.ndf)
        self.decoder_block4 = conv_block_unet(self.ndf * 2, self.ndf, 3, 1, 1)
        self.classifier_conv = nn.Conv2d(self.ndf, self.num_output_channels, 3, 1, 1, 1)

    def forward(self, x):
        s1 = self.encoder_block1(x)
        out = self.maxpool(s1)
        s2 = self.encoder_block2(out)
        out = self.maxpool(s2)
        s3 = self.encoder_block3(out)
        out = self.maxpool(s3)
        s4 = self.encoder_block4(out)
        out = self.maxpool(s4)
        out = self.bottleneck(out)
        out = self.decoder_upsample1(out)
        out = self.decoder_upconv1(out)
        out = torch.cat((out, s4), 1)
        out = self.decoder_block1(out)
        out = self.decoder_upsample2(out)
        out = self.decoder_upconv2(out)
        out = torch.cat((out, s3), 1)
        out = self.decoder_block2(out)
        out = self.decoder_upsample3(out)
        out = self.decoder_upconv3(out)
        out = torch.cat((out, s2), 1)
        out = self.decoder_block3(out)
        out = self.decoder_upsample4(out)
        out = self.decoder_upconv4(out)
        out = torch.cat((out, s1), 1)
        out = self.decoder_block4(out)
        out = self.classifier_conv(out)
        return out


class AEncoder(nn.Module):
    def __init__(self, in_channel=3, height=256, width=256, ndf=16, num_output_channels=8):
        super().__init__()
        self.unet = UNet(in_channel, height, width, ndf, num_output_channels)

    def forward(self, x):
        return torch.tanh(self.unet(x))


class MEncoder(nn.Module):
    def __init__(self, z_length=16, in_channel=3, img_size=256):
        super().__init__()
        self.z_length = z_length
        self.in_channel = in_channel
        self.img_size = img_size

        self.block1 = conv_bn_lrelu(self.in_channel, 16, 3, 2, 1)
        self.block2 = conv_bn_lrelu(16, 32, 3, 2, 1)
        self.block3 = conv_bn_lrelu(32, 64, 3, 2, 1)
        self.block4 = conv_bn_lrelu(64, 128, 3, 2, 1)
        self.fc = nn.Linear(128 * (self.img_size // 16) ** 2, 32)
        self.norm = nn.BatchNorm1d(32)
        self.activ = nn.LeakyReLU(0.03, inplace=True)
        self.mu = nn.Linear(32, self.z_length)
        self.logvar = nn.Linear(32, self.z_length)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x):
        return self.mu(x), self.logvar(x)

    def forward(self, img):
        out = self.block1(img)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.fc(out.view(out.size(0), -1))
        out = self.norm(out)
        out = self.activ(out)
        mu, logvar = self.encode(out)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar


class Segmentor(nn.Module):
    def __init__(self, num_output_channels=8, num_class=1):
        super().__init__()
        self.conv1 = conv_bn_lrelu(num_output_channels, 16, 3, 1, 1)
        self.conv2 = conv_bn_lrelu(16, 16, 1, 1, 0)
        self.pred = nn.Conv2d(16, num_class, 1, 1, 0)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.pred(out)
        return out


class CDDSAModel(nn.Module):
    def __init__(self, z_length=16, in_channel=3, img_size=256, anatomy_channel=8, num_classes=1):
        super().__init__()
        self.m_encoder = MEncoder(z_length=z_length, in_channel=in_channel, img_size=img_size)
        self.a_encoder = AEncoder(in_channel=in_channel, height=img_size, width=img_size, ndf=16, num_output_channels=anatomy_channel)
        self.segmentor = Segmentor(num_output_channels=anatomy_channel, num_class=num_classes)
        self.decoder = Ada_Decoder(anatomy_out_channel=anatomy_channel, z_length=z_length, out_channel=in_channel)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)