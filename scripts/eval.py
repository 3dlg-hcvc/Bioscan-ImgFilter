import matplotlib.pyplot as plt 
import torch 
from torch.utils.data import DataLoader
from dataloaders import get_data_loaders, get_datasets
from trainingFunctions import initialize_model

# Directories
train_dir = "./data/data_splits/train"
val_dir = "./data/data_splits/val"

# Initialize model
model, device = initialize_model()

# Get datasets
train_data, val_data = get_datasets(train_dir, val_dir)

# Create DataLoader for validation dataset
val_loader = DataLoader(val_data, batch_size=4, shuffle=False)

def evaluate(val_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data in val_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.unsqueeze(1).float().to(device)
            
            outputs = torch.sigmoid(model(inputs))
            predicted = (outputs > 0.5).float()

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f'Overall Accuracy: {accuracy * 100:.2f}%')

    return accuracy

def eval_single_sample(val_data):
    idx = torch.randint(0, len(val_data), (1,)).item()
    sample, target = val_data[idx]
    sample = torch.unsqueeze(sample, dim=0).to(device)

    prediction = torch.sigmoid(model(sample))

    if prediction < 0.5:
        print("Prediction: Bad Image")
    else:
        print("Prediction: Good Image")

    # Convert the image tensor to numpy array and permute dimensions
    image_np = sample.squeeze().permute(1, 2, 0).cpu().numpy()

    plt.imshow(image_np)
    plt.title(f"Prediction: {'Good' if prediction > 0.5 else 'Bad'} Image")
    plt.axis('off')
    plt.show()

# Evaluate the model on the validation set
evaluate(val_loader)

# Evaluate a single sample
eval_single_sample(val_data)
