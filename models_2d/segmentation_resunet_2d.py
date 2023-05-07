import torch
import torch.nn as nn
import torchvision
from torchsummary import summary
from torchvision.models import resnet50


class ResNet4Channel(nn.Module):
    def __init__(self):
        super(ResNet4Channel, self).__init__()
        resnet = resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
        weights = resnet.conv1.weight.clone()

        # h x w x 4
        self.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1.weight = nn.Parameter(
            torch.cat((weights, weights[:, 1:2, :, :]), dim=1)
        )
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        # h x 4
        self.layer1 = resnet.layer1  # Bottleneck-3   [-1, 256, 128, 128]
        self.layer2 = resnet.layer2  # Bottleneck-78  [-1, 512, 64, 64]
        self.layer3 = resnet.layer3  # Bottleneck-140 [-1, 1024, 32, 32]
        self.layer4 = resnet.layer4  # Bottleneck172: [-1, 2048, 16, 16]

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x1, x2, x3, x4


class ConvNormRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, padding=padding
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class UpConv(nn.Module):
    def __init__(
        self, in_channels, out_channels, in_channels_up=None, out_channels_up=None
    ):
        super().__init__()
        if in_channels_up is None:
            in_channels_up = in_channels
        if out_channels_up is None:
            out_channels_up = out_channels
        self.upconv = nn.ConvTranspose2d(
            in_channels_up, out_channels_up, kernel_size=2, stride=2
        )
        self.conv1 = ConvNormRelu(in_channels, out_channels)
        self.conv2 = ConvNormRelu(out_channels, out_channels)

    def forward(self, x, x_cat):
        x = self.upconv(x)
        # print("x.shape:", x.shape)
        # print("x_cat.shape:", x_cat.shape)
        x = torch.cat([x, x_cat], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class ResUNet(nn.Module):
    def __init__(self, num_classes):
        super(ResUNet, self).__init__()

        self.resnet = ResNet4Channel()
        self.bottleneck = self._make_bottle_neck(2048, 2048)
        self.pad = nn.ZeroPad2d(
            (0, 0, 0, 1)
        )  # resnet_layer3: [1, 1024, 15, 20]; bottleneck: [1, 2048, 16, 20]
        self.upconv3 = UpConv(2048, 1024)
        self.upconv2 = UpConv(1024, 512)
        self.upconv1 = UpConv(512, 256)
        self.last_conv = nn.Conv2d(256, num_classes, kernel_size=1, stride=1)

    def _make_bottle_neck(self, in_channels, out_channels):
        return nn.Sequential(
            ConvNormRelu(in_channels, out_channels),
            ConvNormRelu(out_channels, out_channels),
        )

    def forward(self, x):
        x1, x2, x3, x4 = self.resnet.forward(x)
        # print("x1.shape:", x1.shape)
        # print("x2.shape:", x2.shape)
        # print("x3.shape:", x3.shape)
        # print("x4.shape:", x4.shape)
        x = self.bottleneck(x4)
        x = self.upconv3.forward(x, x3)
        x = self.upconv2.forward(x, x2)
        x = self.upconv1.forward(x, x1)
        x = self.last_conv(x)
        # x.squeeze(1)
        return x

    def summary(self, input_size):
        summary(self, input_size)


if __name__ == "__main__":
    model = ResUNet(895)
    x = model.forward(torch.randn(1, 4, 240, 320))
    print(x.shape)
    summary(model, (4, 256, 256))
