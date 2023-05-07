import os
import random
import sys
from dataclasses import dataclass
from multiprocessing import cpu_count
from typing import Tuple, Union

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
from pytorch_lightning import Trainer
from skimage import color

sys.path.append("../")
from datasets.nyu_depth_v2 import NYUDepthV2Dataset
from models_2d.segmentation_resunet_2d import ResUNet
from utils.transform_2d import RGBDTransform


@dataclass
class Hyperparameters:
    resize_in: Union[bool, float] = 0.5
    resize_out: Union[None, Tuple[int, int]] = None
    num_epochs: int = 20
    batch_size: int = 4
    num_workers: int = cpu_count()
    learning_rate: float = 0.001

    # Will be set during initialization
    num_classes: int = 895
    mask_shape = None
    rgbd_shape = None
    model_pred_shape = None


@dataclass
class Config:
    data_path: str = os.path.join(os.pardir, ".data/nyu_depth_v2_labeled.mat")
    model_path: str = os.path.join(os.pardir, ".models_2d")
    model_ident: str = "resResUNet_50"
    num_samples: int = 100  # -1 for all
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    random_seed: int = 42
    verbose: bool = True
    print_model: bool = False


class LitResResUNet(pl.LightningModule):
    def __init__(self):
        super(LitResResUNet, self).__init__()

        self.params = Hyperparameters()
        self.config = Config()
        self.verbose = self.config.verbose

        self._init_dataset()

        self.model = ResUNet(num_classes=self.params.num_classes)

        self._set_pred_shape()
        assert self.params.model_pred_shape[2:] == self.params.mask_shape[1:]

        # Criterion, optimizer
        # TODO Try mIoU-Loss
        self.metric_iou = torchmetrics.JaccardIndex(
            task="multiclass", num_classes=self.params.num_classes
        )
        self.criterion = nn.CrossEntropyLoss()

        if self.verbose:
            print(self)
        if self.config.print_model:
            self.model_summary()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        rgbds, masks = batch
        preds = self.model(rgbds)

        loss = self.criterion(preds, masks)
        iou = self.metric_iou(preds, masks)

        self.log("train_loss", loss)
        self.log("train_iou", iou)

        return {"loss": loss}

    # def setup(self, stage=None):  # TODO will be called to often?
    def _init_dataset(self):
        dataset = NYUDepthV2Dataset(
            data_path=self.config.data_path,
            transform=RGBDTransform,  # mean, std and resize will be set in the dataset class
            resize=self.params.resize_in,
            random_seed=self.config.random_seed,
            num_samples=self.config.num_samples,
        )
        self.params.mask_shape = dataset.mask_shape
        self.params.rgbd_shape = dataset.rgbd_shape
        if self.verbose:
            print(dataset)
        self.params.num_classes = dataset.num_classes
        self.train_set, self.val_set = dataset.split_dataset()

    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader(
            dataset=self.train_set,
            batch_size=self.params.batch_size,
            num_workers=self.params.num_workers,
            shuffle=True,
        )
        return train_loader

    def val_dataloader(self):
        val_loader = torch.utils.data.DataLoader(
            dataset=self.val_set,
            batch_size=self.params.batch_size,
            num_workers=self.params.num_workers,
            shuffle=False,
        )
        return val_loader

    def validation_step(self, batch, batch_idx):
        rgbds, masks = batch
        preds = self.model(rgbds)
        loss = self.criterion(preds, masks)
        iou = self.metric_iou(preds, masks)
        self.log("val_loss", loss)
        self.log("val_iou", iou)
        return {"val_loss": loss}

    def on_validation_epoch_end(self, outputs):
        # outputs = list of dictionaries
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {"avg_val_loss": avg_loss}
        # use key 'log'
        return {"val_loss": avg_loss, "log": tensorboard_logs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.params.learning_rate)

    def model_summary(self):
        return super().__repr__()

    def __repr__(self):
        return f"""
    <<<<<<<<<<<<<<<<<<<<PyTorch Lightning Model>>>>>>>>>>>>>>>>>>>>>>
    -+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
    {self.config}
    {self.params}
    """

    def show_segmented(self):
        """
        Display a random example of predicted segmentation and ground truth.
        """
        val_loader = self.val_dataloader()
        rand_idx = random.randint(0, len(val_loader) - 1)
        rgbd, label = list(val_loader)[rand_idx]

        self.model.eval()
        with torch.no_grad():
            pred = self.model(rgbd)[0, :, :, :]
            pred = pred.argmax(0).squeeze(0).cpu().numpy()

        label = label[0, :, :].squeeze(0).numpy()

        fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(18, 6))
        fig.suptitle(f"Sample {rand_idx}")
        ax1.imshow(color.label2rgb(pred))
        ax1.set_title("Predicted segmentation")
        ax1.set_axis_off()
        ax2.imshow(color.label2rgb(label))
        ax2.set_title("Ground truth")
        ax2.set_axis_off()

        plt.show()

    def _set_pred_shape(self):  # TODO fix
        self.model.eval()
        with torch.no_grad():
            x = self.model.forward(torch.randn(1, *self.params.rgbd_shape))
            self.params.model_pred_shape = x.shape

    # def iou_loss(self, preds, labels):
    #     iou = self.train_iou(preds.argmax(dim=1), labels)
    #     return 1 - iou


if __name__ == "__main__":
    model = LitResResUNet()
    # gpus=8
    # fast_dev_run=True -> runs single batch through training and validation
    # train_percent_check=0.1 -> train only on 10% of data
    trainer = Trainer(
        max_epochs=model.params.num_epochs, fast_dev_run=True, log_every_n_steps=25
    )
    trainer.fit(model)

    # advanced features
    # distributed_backend
    # (DDP) implements data parallelism at the module level which can run across multiple machines.
    # 16 bit precision
    # log_gpu_memory
    # TPU support

    # auto_lr_find: automatically finds a good learning rate before training
    # deterministic: makes training reproducable
    # gradient_clip_val: 0 default
