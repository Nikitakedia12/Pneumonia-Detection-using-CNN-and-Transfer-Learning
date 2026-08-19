import sys
import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Add root directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config

def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize(config.IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(config.MEAN, config.STD)
    ])

    eval_transform = transforms.Compose([
        transforms.Resize(config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(config.MEAN, config.STD)
    ])
    
    return train_transform, eval_transform

def get_dataloaders(batch_size=config.BATCH_SIZE):
    train_transform, eval_transform = get_transforms()

    train_dataset = datasets.ImageFolder(root=config.TRAIN_DIR, transform=train_transform)
    val_dataset = datasets.ImageFolder(root=config.VAL_DIR, transform=eval_transform)
    test_dataset = datasets.ImageFolder(root=config.TEST_DIR, transform=eval_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader, train_dataset.classes
