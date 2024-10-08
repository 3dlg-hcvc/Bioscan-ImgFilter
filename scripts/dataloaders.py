import torch
from torchvision import datasets, transforms


def get_datasets(train_dir, val_dir):
    # Define the transformations
    train_transforms = transforms.Compose(
        [   
            # augment the training data
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(30),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    val_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Create datasets
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    val_data = datasets.ImageFolder(val_dir, transform=val_transforms)

    return train_data, val_data


def get_data_loaders(train_data, val_data, batch_size):
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_data, shuffle=True, batch_size=batch_size
    )
    val_loader = torch.utils.data.DataLoader(
        val_data, shuffle=True, batch_size=batch_size
    )

    return train_loader, val_loader



# Define the directories containing the data
train_dir = "./dataset/data_splits/train"
val_dir = "./dataset/data_splits/val"

# Get datasets
train_data, val_data = get_datasets(train_dir, val_dir)

