import random

import torch
import torchvision.transforms as transforms


class RGBDAugmentation:
    """
    https://pytorch.org/vision/stable/transforms.html
    """

    def __init__(self):
        self.horizontal_flip = transforms.RandomHorizontalFlip(p=1.0)
        self.vertical_flip = transforms.RandomVerticalFlip(p=1.0)
        self.rotation = transforms.RandomRotation(degrees=(-10, 10), fill=(0,))
        self.random_crop = transforms.RandomCrop(size=None, padding=4)

    def __call__(self, train_set, augment_ratio=0.5):
        """
        @param train_set: tuple (X_train, y_train)
            X_train: tensor (#Samples, 4, Height, Width)
            y_train: tensor (#Samples, Height, Width)
        @param augment_ratio: float
          @note: the ratio of augmented samples to the original samples
        @return augmented_train_set: tuple (X_train_augmented, y_train_augmented)
        @brief: augment the entir
        """
        X_train, y_train = train_set
        num_samples = X_train.size(0)
        num_augmented_samples = int(augment_ratio * num_samples)

        augmented_X_train = torch.empty((num_augmented_samples, *X_train.size()[1:]))
        augmented_y_train = torch.empty((num_augmented_samples, *y_train.size()[1:]))

        for i in range(num_augmented_samples):
            image = X_train[i, :3]
            depth = X_train[i, 3]
            label = y_train[i]

            if random.random() < 0.5:
                image = self.horizontal_flip(image)
                depth = self.horizontal_flip(depth.unsqueeze(0)).squeeze(0)
                label = self.horizontal_flip(label)

            if random.random() < 0.5:
                image = self.vertical_flip(image)
                depth = self.vertical_flip(depth.unsqueeze(0)).squeeze(0)
                label = self.vertical_flip(label)

            image = self.rotation(image)
            depth = self.rotation(depth.unsqueeze(0)).squeeze(0)
            label = self.rotation(label)

            image = self.random_crop(image)
            depth = self.random_crop(depth.unsqueeze(0)).squeeze(0)
            label = self.random_crop(label)

            augmented_X_train[i] = torch.cat((image, depth.unsqueeze(0)), dim=0)
            augmented_y_train[i] = label

        X_train_augmented = torch.cat((X_train, augmented_X_train), dim=0)
        y_train_augmented = torch.cat((y_train, augmented_y_train), dim=0)

        return X_train_augmented, y_train_augmented
