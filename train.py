import logging
import zipfile
import os

from torchvision import transforms
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from helpers import kaggle


logging.basicConfig(
    filename="train.log",
    filemode="a",
    format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)

torch.manual_seed(42)

IMAGE_SIZE = (128, 128)
COLOR = "RGB"  # 'L' or 'RGB'
CHANNELS = 3 if COLOR == "RGB" else 1
BATCH_SIZE = 64
EPOCH = 30


def download_dataset():
    """
    This function downloads the dataset from Kaggle and extracts it to the data folder.
    """

    if os.path.exists("data"):
        return

    kaggle.api.dataset_download_files("shaunthesheep/microsoft-catsvsdogs-dataset")
    with zipfile.ZipFile("microsoft-catsvsdogs-dataset.zip", "r") as zip:
        zip.extractall("data")
    os.remove("microsoft-catsvsdogs-dataset.zip")
    # remove dog 11702, cat 666 (corrupted images)
    os.remove("data/PetImages/Cat/666.jpg")
    os.remove("data/PetImages/Dog/11702.jpg")


def load_dataset():
    transform = transforms.Compose(
        [
            transforms.Lambda(lambda img: img.convert(COLOR)),
            transforms.Resize(IMAGE_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5] * CHANNELS, std=[0.5] * CHANNELS),
        ]
    )

    dataset = ImageFolder("data/PetImages", transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = (len(dataset) - train_size) // 2
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader


class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            # 1st conv layer
            nn.Conv2d(CHANNELS, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # 2nd conv layer
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # 3rd conv layer
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # 4th conv layer
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # 5th conv layer
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # 6th conv layer
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # Flatten
            nn.Flatten(),
            # 1st dense layer
            nn.Linear(256 * IMAGE_SIZE[0] // 16 * IMAGE_SIZE[1] // 16, 512),
            nn.BatchNorm1d(512),
            # 2nd dense layer
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            # Output layer
            nn.Linear(128, 2),
        )

    def forward(self, x):
        x = self.network(x)
        return x


@torch.no_grad()
def evaluate(net, loader):
    sum_loss = 0
    loss = nn.CrossEntropyLoss()

    net.train(False)
    for data in loader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)
        sum_loss += loss(outputs, labels).item()

    net.train(True)
    return sum_loss / len(loader)


@torch.no_grad()
def accuracy(net, loader):
    correct = 0
    total = 0

    net.train(False)
    for data in loader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    net.train(True)
    return correct / total


if __name__ == "__main__":
    download_dataset()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, val_loader, test_loader = load_dataset()

    net = Net().to(device)
    logging.info(
        f"Model parameters: {sum(p.numel() for p in net.parameters() if p.requires_grad)}"
    )
    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    for epoch in range(EPOCH):
        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            l = loss(outputs, labels)
            l.backward()
            optimizer.step()

            if i % 100 == 0:
                logging.info(
                    f"Epoch {epoch}, batch {i}, val_loss: {evaluate(net, val_loader):.5f}"
                )

    logging.info(f"Test loss: {evaluate(net, test_loader):.5f}")
    logging.info(f"Test accuracy: {accuracy(net, test_loader):.5f}")
    torch.save(net.state_dict(), "model.pth")