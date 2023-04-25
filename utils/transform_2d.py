from typing import Tuple, Union

import numpy as np
import torch
import torchvision.transforms as transforms


class RGBDTransform:
    def __init__(self, resize=Union[None, Tuple[int, int]], mean=None, std=None):
        self.resize = resize
        self.mean = mean if mean else [0.485, 0.456, 0.406, 0.5]
        self.std = std if std else [0.229, 0.224, 0.225, 0.5]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.compose = transforms.Compose(
            [
                transforms.Resize(size=self.resize),
                transforms.Normalize(mean=self.mean, std=self.std),
            ]
        )

    def __call__(self, sample):
        image, depth, label = sample
        # assert (
        #     image.shape[:3] == depth.size[:3]
        # ), "Image and depth map must have the same size"

        # Concatenate the depth channel as the fourth channel
        rgbd = torch.cat((image, depth.unsqueeze(-1)), dim=-1)

        # rgbd = self.compose(rgbd)
        print(rgbd.shape)

        # Move label to GPU if available
        # label = torch.tensor(label, dtype=torch.long).to(self.device)

        return rgbd, label
