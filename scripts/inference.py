import torch
import torchvision.transforms as T
import os
import sys

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../data_processing"))
)
from dataloaders import get_datasets
from training_helperFunctions import initialize_model

# Directories
train_dir = "./dataset/data_splits/train"
val_dir = "./dataset/data_splits/val"

# Initialize model
model, device = initialize_model()

# Get datasets
train_data, val_data = get_datasets(train_dir, val_dir)


# Evaluate a single sample from the validation data
def eval_single_sample(val_data, model, device):
    model.eval()  # Set the model to evaluation mode

    # Ensure no gradients are calculated during inference
    with torch.no_grad():
        idx = torch.randint(0, len(val_data), (1,)).item()
        sample = torch.unsqueeze(val_data[idx][0], dim=0).to(device)

        prediction = torch.sigmoid(model(sample))
        prediction_value = prediction.item() * 100

        if prediction < 0.5:
            print("Prediction: Bad Image")
            print(f"Percent Bad: {prediction_value:.2f}%")

        else:
            print("Prediction: Good Image")
            print(f"Percent good: {prediction_value:.2f}%")

        # Transform tensor to PIL image
        transform = T.ToPILImage()
        img = transform(
            sample.squeeze(0).cpu().clamp(0, 1)
        )  # Remove the batch dimension, move to CPU, and clamp values to [0, 1]

        img.show()


# Calling the function to infer whether an image is good/bad
eval_single_sample(val_data, model, device)
