import os
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from data.preprocessing import get_transforms
from config.config import BATCH_SIZE

def load_datasets(dataset_path):

    train_transform, test_transform = get_transforms()

    train_dir = os.path.join(dataset_path, "train")
    test_dir = os.path.join(dataset_path, "test")

    train_dataset = ImageFolder(train_dir, transform=train_transform)
    test_dataset = ImageFolder(test_dir, transform=test_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    return train_loader, test_loader