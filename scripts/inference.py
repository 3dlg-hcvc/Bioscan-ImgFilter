import matplotlib.pyplot as plt 
import torch 
import torchvision.transforms as T
from dataloaders import get_datasets
from trainingFunctions import initialize_model
from PIL import Image

# Directories
train_dir = "./data/data_splits/train"
val_dir = "./data/data_splits/val"

# Initialize model
model, device = initialize_model()

# Get datasets
train_data, val_data = get_datasets(train_dir, val_dir)

def eval_single_sample(val_data, model, device):
    model.eval()  # Set the model to evaluation mode
    
    # Ensure no gradients are calculated during inference
    with torch.no_grad():
        idx = torch.randint(0, len(val_data), (1,)).item()
        sample = torch.unsqueeze(val_data[idx][0], dim=0).to(device)
        
        prediction = torch.sigmoid(model(sample))

        if prediction < 0.5:
            print("Prediction: Bad Image")
        else:
            print("Prediction: Good Image")
        
        # Transform tensor to PIL image
        transform = T.ToPILImage()
        img = transform(sample.squeeze(0).cpu().clamp(0, 1))  # Remove the batch dimension, move to CPU, and clamp values to [0, 1]
        
        img.show()

# Evaluate a single sample from the validation data
eval_single_sample(val_data, model, device)
