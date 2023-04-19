import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image
from torchvision.transforms import Normalize, Resize, ToTensor


class RGBDTransform:
    def __init__(self, resize=None, normalize=True, mean=None, std=None):
        self.resize = resize
        self.normalize = normalize
        self.mean = mean if mean else [0.485, 0.456, 0.406, 0.5]
        self.std = std if std else [0.229, 0.224, 0.225, 0.5]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __call__(self, sample):
        image = sample["rgb_image"]
        depth = sample["depth_map"]
        label = sample["label"]

        assert image.size == depth.size, "Image and depth map must have the same size"

        if self.resize:
            image = F.resize(image, self.resize, interpolation=Image.BILINEAR)
            depth = F.resize(depth, self.resize, interpolation=Image.NEAREST)

        image = np.array(image, np.float32) / 255.0
        depth = np.array(depth, np.float32) / 255.0

        # Concatenate the depth channel as the fourth channel
        rgbd = np.dstack((image, depth))

        if self.normalize:
            rgbd = (rgbd - self.mean) / self.std

        rgbd = ToTensor()(rgbd).to(self.device)

        # Move label to GPU if available
        label = torch.tensor(label, dtype=torch.long).to(self.device)

        return rgbd, label
