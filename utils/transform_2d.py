from typing import List, Tuple, Union

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch.transforms import ToTensorV2


class RGBDTransform:
    """ """

    def __init__(
        self,
        resize: Union[None, Tuple[int, int]],
        mean: Union[None, List[float]] = None,
        std: Union[None, List[float]] = None,
    ):
        # default: 480, 640 for NYUv2
        self.height, self.width = resize
        self.mean = mean if mean else [0.485, 0.456, 0.406, 0.5]
        self.std = std if std else [0.229, 0.224, 0.225, 0.5]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tf_resize_rgbd = A.Resize(
            width=self.width, height=self.height, interpolation=cv2.INTER_LINEAR
        )
        self.tf_resize_mask = A.Resize(
            width=self.width, height=self.height, interpolation=cv2.INTER_NEAREST
        )

        self.tf_augment_rgb = A.Compose(
            [
                A.RGBShift(
                    r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.95
                ),
                A.OneOf(
                    [
                        A.Blur(blur_limit=3, p=0.5),
                        A.ColorJitter(p=0.8),
                    ],
                    p=0.8,
                ),
                A.RandomBrightnessContrast(p=0.6),
                A.GaussianBlur(p=0.2),
            ]
        )
        self.tf_augment = A.Compose(
            [
                A.RandomCrop(
                    width=int(self.width * 0.7), height=int(self.height * 0.7), p=0.45
                ),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.RandomScale(scale_limit=0.2, p=1),
                A.ElasticTransform(p=0.7, alpha=32, sigma=50, alpha_affine=120 * 0.03),
                A.GridDistortion(num_steps=1, distort_limit=0.2, p=0),
                A.ShiftScaleRotate(
                    shift_limit=0.0625, scale_limit=0.05, rotate_limit=15, p=0
                ),
                A.OneOf(
                    [
                        A.Cutout(
                            num_holes=8,
                            max_h_size=16,
                            max_w_size=16,
                            fill_value=0,
                            p=0.5,
                        ),
                        A.CoarseDropout(
                            max_holes=8, max_height=16, max_width=16, p=0.5
                        ),
                    ],
                    p=0.2,
                ),
            ],
        )
        self.tf_normalize = A.Compose(
            [
                A.Normalize(mean=self.mean, std=self.std, max_pixel_value=1.0),
                ToTensorV2(),
            ]
        )
        self.tf_resize_mask_final = A.Compose(
            [
                A.Resize(
                    width=self.width // 4,
                    height=self.height // 4,
                    interpolation=cv2.INTER_NEAREST,
                ),
                ToTensorV2(),
            ]
        )

    def __call__(self, rgb, depth, mask, split_type):
        rgb = self.tf_resize_rgbd(image=rgb)["image"]
        depth = self.tf_resize_rgbd(image=depth)["image"]
        mask = self.tf_resize_mask(image=mask)["image"]  # use INTER_NEAREST

        # augment
        if split_type == "train":
            rgb = self.tf_augment_rgb(image=rgb)["image"]
            # Concatenate the depth channel as the fourth channel 3xHxW, HxW -> 4xHxW
            rgb = rgb.astype(np.float32) / 255.0
            rgbd = np.concatenate((rgb, np.expand_dims(depth, axis=-1)), axis=-1)
            augmented = self.tf_augment(image=rgbd, mask=mask)
            rgbd, mask = augmented["image"], augmented["mask"]
        else:
            rgb = rgb.astype(np.float32) / 255.0
            rgbd = np.concatenate((rgb, np.expand_dims(depth, axis=-1)), axis=-1)

        rgbd = self.tf_normalize(image=rgbd)["image"]

        mask = self.tf_resize_mask_final(image=mask)["image"]

        return rgbd.to(self.device), mask.to(self.device)
