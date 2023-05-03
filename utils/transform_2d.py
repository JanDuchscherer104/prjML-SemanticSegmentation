from typing import Tuple, Union

import torch
import torchvision.transforms as transforms


class RGBDTransform:
    """
    https://pytorch.org/vision/stable/transforms.html
    TODO crop,
    """

    def __init__(self, resize=Union[None, Tuple[int, int]], mean=None, std=None):
        self.resize = resize
        self.mean = mean if mean else [0.485, 0.456, 0.406, 0.5]
        self.std = std if std else [0.229, 0.224, 0.225, 0.5]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.rgbd_transform = transforms.Compose(
            [
                transforms.Resize(
                    size=self.resize,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                    antialias=True,
                ),
                transforms.Normalize(mean=self.mean, std=self.std),
            ]
        )
        self.label_transform = transforms.Compose(
            [
                transforms.Resize(
                    # size=self.resize,
                    size=(120, 160),
                    interpolation=transforms.InterpolationMode.NEAREST,
                )
            ]
        )

    def __call__(self, sample):
        image, depth, label = sample

        # Concatenate the depth channel as the fourth channel 3xhxw, hxw -> 4xhxw
        rgbd = torch.cat((image, depth.unsqueeze(0)), dim=0)
        rgbd = self.rgbd_transform(rgbd).to(self.device)
        label = self.label_transform(label.unsqueeze(0)).squeeze(0).to(self.device)

        return rgbd, label
