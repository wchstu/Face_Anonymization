import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.landmarks_utils import LandmarksHeatMapDecoder

def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

def adaptive_instance_normalization(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)

def create_pyramid(img, n):
    if isinstance(img, (list, tuple)):
        return img
    pyd = [img]
    for i in range(n-1):
        pyd.append(F.avg_pool2d(pyd[-1], 3, stride=2, padding=1, count_include_pad=False))
    return pyd

class ResidualBlockDown(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None):
        super(ResidualBlockDown, self).__init__()

        # Right Side
        self.conv_r1 = ConvLayer(in_channels, out_channels, kernel_size, stride, padding)
        self.conv_r2 = ConvLayer(out_channels, out_channels, kernel_size, stride, padding)

        # Left Side
        self.conv_l = ConvLayer(in_channels, out_channels, 1, 1)

    def forward(self, x):
        residual = x

        # Right Side
        out = F.relu(x)
        out = self.conv_r1(out)
        out = F.relu(out)
        out = self.conv_r2(out)
        out = F.avg_pool2d(out, 2)

        # Left Side
        residual = self.conv_l(residual)
        residual = F.avg_pool2d(residual, 2)

        # Merge
        out = residual + out
        return out

class ResidualBlockUp(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, upsample=2):
        super(ResidualBlockUp, self).__init__()

        # General
        self.upsample = nn.Upsample(scale_factor=upsample, mode='nearest')

        # Right Side
        self.norm_r1 = nn.InstanceNorm2d(in_channels, affine=True)
        self.conv_r1 = ConvLayer(in_channels, out_channels, kernel_size, stride)

        self.norm_r2 = nn.InstanceNorm2d(out_channels, affine=True)
        self.conv_r2 = ConvLayer(out_channels, out_channels, kernel_size, stride)

        # Left Side
        self.conv_l = ConvLayer(in_channels, out_channels, 1, 1)

    def forward(self, x):
        residual = x

        # Right Side
        out = self.norm_r1(x)
        out = F.relu(out)
        out = self.upsample(out)
        out = self.conv_r1(out)
        out = self.norm_r2(out)
        out = F.relu(out)
        out = self.conv_r2(out)

        # Left Side
        residual = self.upsample(residual)
        residual = self.conv_l(residual)

        # Merge
        out = residual + out
        return out


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in1 = nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in2 = nn.InstanceNorm2d(channels, affine=True)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.in1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.in2(out)

        out = out + residual
        return out


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=None):
        super(ConvLayer, self).__init__()
        if padding is None:
            padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(padding)   # 镜像填充
        self.conv2d = nn.utils.spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size, stride))

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out

class Generator(nn.Module):
    def __init__(self, n_input=3, n_emb=512, img_size=128, device=torch.device('cpu')):
        super(Generator, self).__init__()
        self.n_input = n_input
        self.n_emb = n_emb
        self.img_size = img_size
        self.device = device

        self.lnd_128 = nn.Sequential(
            LandmarksHeatMapDecoder(img_size).to(device),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(n_input - 3, 3, 3, 1, padding=0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.InstanceNorm2d(3, affine=True)
        )
        self.lnd_64 = nn.Sequential(
            LandmarksHeatMapDecoder(img_size//2).to(device),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(n_input - 3, 3, 3, 1, padding=0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.InstanceNorm2d(3, affine=True)
        )
        self.lnd_32 = nn.Sequential(
            LandmarksHeatMapDecoder(img_size // 4).to(device),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(n_input - 3, 3, 3, 1, padding=0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.InstanceNorm2d(3, affine=True)
        )

        self.down_128 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=6, out_channels=32, kernel_size=3, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.InstanceNorm2d(32, affine=True),
            ResidualBlockDown(32, 64),
            nn.InstanceNorm2d(64, affine=True)
        )

        self.down_64_1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=6, out_channels=64, kernel_size=3, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.InstanceNorm2d(64, affine=True)
        )
        self.down_64_2 = nn.Sequential(
            ResidualBlockDown(128, 128),
            nn.InstanceNorm2d(128, affine=True)
        )

        self.down_32_1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=6, out_channels=128, kernel_size=3, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.InstanceNorm2d(128, affine=True)
        )
        self.down_32_2 = nn.Sequential(
            ResidualBlockDown(256, 256),
            nn.InstanceNorm2d(256, affine=True)
        )

        self.down_16 = nn.Sequential(
            ResidualBlockDown(256, 512),
            nn.InstanceNorm2d(512, affine=True)
        )

        self.convert_1 = nn.Sequential(
            nn.Linear(self.n_emb, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 2048),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.convert_2 = nn.Sequential(
            ConvLayer(32, 64, kernel_size=3, stride=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.InstanceNorm2d(64, affine=True),
            ConvLayer(64, 128, kernel_size=3, stride=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.InstanceNorm2d(128, affine=True),
            ConvLayer(128, 256, kernel_size=3, stride=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.InstanceNorm2d(256, affine=True),
            ConvLayer(256, 512, kernel_size=3, stride=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.InstanceNorm2d(512)
        )
        self.convert_3 = nn.Sequential(
            ConvLayer(1024, 512, kernel_size=3, stride=1),
            nn.InstanceNorm2d(512, affine=True),
            ResidualBlock(512)
        )

        self.up_8_2 = nn.Sequential(
            ResidualBlockUp(1024, 256, upsample=2),
            nn.InstanceNorm2d(256, affine=True)
        )

        self.up_16_2 = nn.Sequential(
            ResidualBlockUp(512, 128, upsample=2),
            nn.InstanceNorm2d(128, affine=True)
        )

        self.up_32_2 = nn.Sequential(
            ResidualBlockUp(256, 64, upsample=2),
            nn.InstanceNorm2d(64, affine=True)
        )

        self.up_64_2 = nn.Sequential(
            ResidualBlockUp(128, 32, upsample=2),
            nn.InstanceNorm2d(32, affine=True)
        )

        self.up_128 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(32, 3, 3, 1, padding=0),
            nn.Tanh()
        )

    def forward(self, x, lndm, embeddings):
        pyds = create_pyramid(x, n=3)

        down_64 = self.down_128(torch.cat((pyds[0], self.lnd_128(lndm)), dim=1))
        out = torch.cat((down_64, self.down_64_1(torch.cat((pyds[1], self.lnd_64(lndm)), dim=1))), dim=1)
        down_32 = self.down_64_2(out)
        out = torch.cat((down_32, self.down_32_1(torch.cat((pyds[2], self.lnd_32(lndm)), dim=1))), dim=1)
        down_16 = self.down_32_2(out)
        down_8 = self.down_16(down_16)

        face_embd = self.convert_1(embeddings)
        face_embd = self.convert_2(face_embd.view(-1, 32, 8, 8))
        out = torch.cat((down_8, face_embd), dim=1)
        out = self.convert_3(out)

        # out = adaptive_instance_normalization(out, features[0])
        out = torch.cat((out, down_8), dim=1)
        out = self.up_8_2(out)
        # out = adaptive_instance_normalization(out, features[1])
        out = torch.cat((out, down_16), dim=1)
        out = self.up_16_2(out)
        # out = adaptive_instance_normalization(out, features[2])
        out = torch.cat((out, down_32), dim=1)
        out = self.up_32_2(out)
        # out = adaptive_instance_normalization(out, features[3])
        out = torch.cat((out, down_64), dim=1)
        out = self.up_64_2(out)
        out = self.up_128(out)

        return out

if __name__ == '__main__':
    batch = 16
    model = Generator(n_input=101)
    x = torch.randn(batch, 3, 128, 128)
    lndm = torch.randn(batch, 98, 2)
    features = [torch.randn(batch, 512, 8, 8),
                torch.randn(batch, 256, 16, 16),
                torch.randn(batch, 128, 32, 32),
                torch.randn(batch, 64, 64, 64)]
    embedding = torch.randn(batch, 512)
    y = model(x, lndm, embedding)
    print(y.shape)

    '''
    torch.Size([1, 512, 8, 8])
    torch.Size([1, 256, 16, 16])
    torch.Size([1, 128, 32, 32])
    torch.Size([1, 64, 64, 64])
    '''