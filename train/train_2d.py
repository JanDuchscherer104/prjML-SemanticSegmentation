import os
import random
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from skimage import color
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append("../")
from datasets.nyu_depth_v2 import NYUDepthV2Dataset
from models_2d.segmentation_unet_2d import UNet
from utils.transform_2d import RGBDTransform


class ResUNetTrainer:
    MODEL_DIR = ".models_2d/"

    def __init__(self, model_ident="resunet_50"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = os.path.join(
            os.getcwd(), os.pardir, self.MODEL_DIR, f"{model_ident}.pth"
        )
        self.model = None

    def get_data_loaders(self, data_path, batch_size=8, num_samples=-1):
        dataset = NYUDepthV2Dataset(
            data_path=data_path,
            transform=RGBDTransform,
            resize=False,
            num_samples=num_samples,
        )
        print(dataset)
        self.num_classes = dataset.num_classes
        train_set, val_set = dataset.split_dataset()

        self.train_loader = DataLoader(
            train_set, batch_size=batch_size, shuffle=True, num_workers=2
        )
        self.val_loader = DataLoader(
            val_set, batch_size=batch_size, shuffle=False, num_workers=2
        )

    def train(self, batch_size=8, num_epochs=1, learning_rate=1e-4):
        if self.model is None:
            self.load_model()

        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate

        # Model, Loss, and Optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # Training and Validation loop
        past_epochs = 0
        try:
            for epoch in tqdm(range(num_epochs)):
                self.model.train()
                train_loss = 0
                for rgbd, labels in self.train_loader:
                    optimizer.zero_grad()
                    preds = self.model(rgbd)
                    loss = criterion(preds, labels)
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()
                    # add mIoU
                    print(f"Train Loss: {train_loss:.4f}", end="\r")

                train_loss /= len(self.train_loader)

                self.model.eval()
                val_loss = 0
                with torch.no_grad():
                    for rgbd, labels in self.val_loader:
                        preds = self.model(rgbd)
                        loss = criterion(preds, labels)

                        val_loss += loss.item()

                val_loss /= len(self.val_loader)

                print(
                    f"Epoch: {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
                )
                past_epochs += 1
        except KeyboardInterrupt:
            print("Interrupted")
            if past_epochs > 5:
                self.save_model()
            return

    def load_model(self, from_file=False):
        self.model = UNet(num_classes=self.num_classes).to(self.device)
        if from_file:
            self.model.load_state_dict(torch.load(self.model_path))
            print(f"Loaded model from {self.model_path}")

    def save_model(self):
        torch.save(self.model.state_dict(), self.model_path)
        print(f"Model saved to {self.model_path}")

    def show_segmented(self):
        rand_idx = random.randint(0, len(self.val_loader) - 1)
        rgbd, label = list(self.val_loader)[rand_idx]

        self.model.eval()
        with torch.no_grad():
            pred = self.model(rgbd)[0, :, :, :]
            pred = pred.argmax(0).squeeze(0).cpu().numpy()

        # rgb_image = rgbd.squeeze(0)[:3:, :, :]
        # print(rgb_image)
        # print(rgb_image.shape)
        # rgb_image = rgb_image.numpy().astype(np.float32).transpose(1, 2, 0)

        label = label[0, :, :].squeeze(0)

        # print(rgb_image.shape)

        fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(18, 6))
        fig.suptitle(f"Sample {rand_idx}")
        # ax0.imshow(rgb_image)
        # ax0.set_title("RGB image")
        # ax0.set_axis_off()
        ax1.imshow(color.label2rgb(pred))
        ax1.set_title("Predicted segmentation")
        ax1.set_axis_off()
        ax2.imshow(color.label2rgb(label.numpy()))
        ax2.set_title("Ground truth")
        ax2.set_axis_off()

        plt.show()


if __name__ == "__main__":
    trainer = ResUNetTrainer()
    trainer.get_data_loaders(
        data_path="/Volumes/Extreme SSD/nyu_depth_v2_labeled.mat",
        num_samples=-1,
    )
    trainer.load_model(from_file=False)
    trainer.train(num_epochs=20)
    trainer.save_model()
    trainer.show_segmented()
