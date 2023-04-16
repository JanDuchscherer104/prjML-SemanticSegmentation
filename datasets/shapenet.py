import json
import os
from copy import deepcopy

import numpy as np
import torch
from torch.utils.data import Dataset


class ShapeNetDataset(Dataset):
    __data_dir = os.path.join(__file__, os.pardir, os.pardir, ".data")
    __set_name = "shapenetcore_partanno_segmentation_benchmark_v0_normal"
    __set_name = os.path.join(__set_name, __set_name)

    __set_info = "synsetoffset2category.txt"

    def __init__(self, data_root=None, split="train", transform=None, verbose=True):
        super().__init__()
        self.data_root = os.path.abspath(
            data_root if data_root else os.path.join(self.__data_dir, self.__set_name)
        )
        self.split = split
        self.transform = transform
        self.verbose = verbose
        self.__set_categories()

    def __set_categories(self):
        with open(os.path.join(self.data_root, self.__set_info), "r") as f:
            self.categories = dict(line.strip().split() for line in f.readlines())
        if self.verbose:
            print("Categories:", self.categories)


if __name__ == "__main__":
    dataset = ShapeNetDataset()
