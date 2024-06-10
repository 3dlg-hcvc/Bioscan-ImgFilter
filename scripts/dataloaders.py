import torch
import torchvision
from torchvision import datasets, transforms

def get_datasets(train_dir, val_dir):
    # Define the transformations
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create datasets
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    print(train_data)
    val_data = datasets.ImageFolder(val_dir, transform=val_transforms)

    return train_data, val_data

def get_data_loaders(train_data, val_data, batch_size):
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=batch_size)
    val_loader = torch.utils.data.DataLoader(val_data, shuffle=True, batch_size=batch_size)

    return train_loader, val_loader

# View the labels (0 or 1) associated with each image 
def inspect_labels(dataset):
    class_to_idx = dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    print("Class to index mapping:", class_to_idx)
    print("\nSample image paths and labels:")
    
    for i in range(100):  # print first 100 samples
        image_path, label = dataset.samples[i]
        print(f"Path: {image_path}, Label: {label} ({idx_to_class[label]})")

# Define the directories containing the data
train_dir = "./data/data_splits/train"
val_dir = "./data/data_splits/val"

# Get datasets
train_data, val_data = get_datasets(train_dir, val_dir)

# Inspect labels
#inspect_labels(train_data)
