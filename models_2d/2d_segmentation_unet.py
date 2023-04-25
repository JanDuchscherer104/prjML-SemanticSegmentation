import torch
import torch.nn as nn
import torchvision
from torchsummary import summary
from torchvision.models import resnet50
from torchvision.models.resnet import Bottleneck

# def __init__(self, pretrained=True):
#     super(Res50, self).__init__()

#     self.de_pred = nn.Sequential(
#         Conv2d(1024, 128, 1, same_padding=True, NL="relu"),
#         Conv2d(128, 1, 1, same_padding=True, NL="relu"),
#     )

#     res.conv1.weight[:, :3, :, :] = torch.nn.Parameter(weights)
#     res.conv1.weight[:, 3, :, :] = torch.nn.Parameter(weights[:, 1, :, :])

#     self.frontend = nn.Sequential(
#         res.conv1, res.bn1, res.relu, res.maxpool, res.layer1, res.layer2
#     )

#     self.own_reslayer_3 = make_res_layer(Bottleneck, 256, 6, stride=1)
#     self.own_reslayer_3.load_state_dict(res.layer3.state_dict())


class ResNet4Channel(nn.Module):
    def __init__(self, out_features=1000):
        super(ResNet4Channel, self).__init__()
        backbone = resnet50(weights=torchvision.models.ResNet50_Weights)
        weights = backbone.conv1.weight.clone()

        # h x w x 4
        self.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1.weight = nn.Parameter(
            torch.cat((weights, weights[:, 1:2, :, :]), dim=1)
        )
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        # h x 4
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        # self.avgpool = backbone.avgpool
        # self.fc = nn.Linear(512 * 4, out_features)

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

    def print_summary(self, input_size):
        summary(self, input_size)


class UNet(nn.Module):
    def® __init__(self, num_classes):
        super(UNet, self).__init__()
        self.resnet = ResNet4Channel()
        # add bottleneck layer
        self.bottleneck = nn.Sequential(
            nn.Conv2d(2048, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.decoder3 = self._make_decoder_layer(256, 128)
        self.decoder2 = self._make_decoder_layer(128, 64)
        self.decoder1 = self._make_decoder_layer(64, 32)
        self.last_conv = nn.Conv2d(32, num_classes, kernel_size=1)

    def _make_decoder_layer(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x1, x2, x3, x4 = self.resnet.forward(x)
        x4 = self.bottleneck(x4)
        x = self.upconv3(x4)
        x = torch.cat([x, x3], dim=1)
        x = self.decoder3(x)
        x = self.upconv2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.decoder2(x)
        x = self.upconv1(x)
        x = torch.cat([x, x1], dim=1)
        x = self.decoder1(x)
        x = self.last_conv(x)
        return x

    def print_summary(self, input_size):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = self.to(device)
        summary(model, input_size)


if __name__ == "__main__":
    model = ResNet4Channel()  # Assuming UNet and num_classes are defined
    input_size = (
        4,
        128,
        128,
    )  # Example input size: 4-channel RGB-D image with 224x224 resolution
    model.print_summary(input_size)
    # model.print_shapes()
