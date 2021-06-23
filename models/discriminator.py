import torch
import torch.nn as nn
import torch.utils.data
from torch.nn import functional as F

class Discriminator(nn.Module):
    def __init__(self, input_nc=3, img_size=64, **kwargs):
        super(Discriminator, self).__init__()

        self.img_size = img_size

        self.conv1 = ResidualBlockDown(input_nc, 64)
        self.conv2 = ResidualBlockDown(64, 128)
        self.conv3 = ResidualBlockDown(128, 256)
        self.conv4 = ResidualBlockDown(256, 512)
        if img_size == 128:
            self.conv5 = ResidualBlockDown(512, 512)

        self.dense0 = nn.Linear(8192, 1024)
        self.dense1 = nn.Linear(1024, 1)

    def forward(self, x):
        out = x  # [B, 6, 64, 64]
        # Encode
        out_0 = (self.conv1(out))  # [B, 64, 32, 32]
        out_1 = (self.conv2(out_0))  # [B, 128, 16, 16]
        out_3 = (self.conv3(out_1))  # [B, 256, 8, 8]
        out = (self.conv4(out_3))  # [B, 512, 4, 4]
        if self.img_size == 128:
            out = (self.conv5(out))  # [B, 512, 4, 4]

        out = out.view(out.size(0), -1)
        out = F.leaky_relu(self.dense0(out), 0.2, inplace=True)
        out = F.leaky_relu(self.dense1(out), 0.2, inplace=True)
        return out


# region Residual Blocks
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
# endregion

if __name__ == '__main__':
    dis = Discriminator(input_nc=3, img_size=128)
    x = torch.randn(8, 3, 128, 128)
    y = dis(x)