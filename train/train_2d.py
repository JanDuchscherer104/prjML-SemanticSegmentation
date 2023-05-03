import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append("../")
from datasets.nyu_depth_v2 import NYUDepthV2Dataset
from models_2d.segmentation_unet_2d import UNet
from utils.transform_2d import RGBDTransform


def main():
    # Hyperparameters
    batch_size = 8
    num_epochs = 50
    learning_rate = 1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = NYUDepthV2Dataset(
        data_path="/Volumes/Extreme SSD/nyu_depth_v2_labeled.mat",
        transform=RGBDTransform,
        resize=False,
    )
    num_classes = dataset.num_classes

    train_set, val_set = dataset.split_dataset()

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=4
    )

    # Model, Loss, and Optimizer
    model = UNet(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training and Validation loop
    for epoch in tqdm(range(num_epochs)):
        model.train()
        train_loss = 0
        for rgbd, labels in train_loader:
            rgbd, labels = rgbd.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(rgbd)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for rgbd, labels in val_loader:
                rgbd, labels = rgbd.to(device), labels.to(device)

                outputs = model(rgbd)
                loss = criterion(outputs, labels)

                val_loss += loss.item()

        val_loss /= len(val_loader)

        print(
            f"Epoch: {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
        )


if __name__ == "__main__":
    main()
