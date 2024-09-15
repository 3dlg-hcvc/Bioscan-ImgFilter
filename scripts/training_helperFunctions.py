import os
from torch.nn.modules.loss import BCEWithLogitsLoss
import torch
import torchvision.models as models
from torchvision.models import ResNet18_Weights
import torchvision.transforms as T

import torch.nn as nn
from PIL import Image
import numpy as np 



# Training step for neural network
def train_step(model, optimizer, loss_fn, input_data, target_labels):

    # enter train mode
    model.train()

    # make prediction
    input_predictions = model(input_data)

    # compute loss
    loss = loss_fn(input_predictions, target_labels)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    return loss


# Initialize a ResNet18 model with a custom final layer.
def initialize_model():

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze last few layers
    for param in model.layer3.parameters():
        param.requires_grad = True

    # for param in model.layer3.parameters():
    #    param.requires_grad = True

    # Add a new final layer
    num_filters = model.fc.in_features
    # model.fc = nn.Linear(num_filters, 1)
    model.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(num_filters, 1))

    # Move model to device
    model = model.to(device)

    return model, device


# Get binary cross-entropy loss function.
def get_loss_fn():
    return BCEWithLogitsLoss()



def misclassified_imgs(input_batch, predicted_labels, target_labels):
    misclassified_dir = "misclassified_imgs"
    os.makedirs(misclassified_dir, exist_ok=True)

    # Convert tensor to PIL image
    transform = T.ToPILImage()

    # Ensure tensors
    # Ensure tensors are on CPU and detached from the graph
    predicted_labels = predicted_labels.clone().detach().cpu()
    target_labels = target_labels.clone().detach().cpu()

    # Determine which images were misclassified
    misclassified_indices = (predicted_labels.squeeze() != target_labels.squeeze()).cpu().numpy()
   

    for i in range(len(misclassified_indices)):
        if misclassified_indices[i]:
            # Convert the misclassified tensor image to a PIL image
            misclassified_image = transform(
                input_batch[i].cpu().clamp(0, 1).squeeze(0)  # Remove the batch dimension
            )

            # Get the true and predicted labels
            true_label = int(target_labels[i].cpu().numpy().item())
            predicted_label = int(predicted_labels[i].cpu().numpy().item())
            file_name = f"idx_{i}_true_{true_label}_pred_{predicted_label}.png"
            save_path = os.path.join(misclassified_dir, file_name)

            # Debugging: Print the save path
            #print(f"Saving misclassified image: {save_path}")

            misclassified_image.save(save_path)
