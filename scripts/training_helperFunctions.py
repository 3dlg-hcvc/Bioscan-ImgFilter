from torch.nn.modules.loss import BCEWithLogitsLoss
import torch
import torchvision.models as models
from torchvision.models import ResNet18_Weights
import torch.nn as nn


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
    for param in model.layer4.parameters():
        param.requires_grad = True

    # for param in model.layer3.parameters():
    #    param.requires_grad = True

    # Add a new final layer
    num_filters = model.fc.in_features
    # model.fc = nn.Linear(num_filters, 1)
    model.fc = nn.Sequential(nn.Dropout(0.6), nn.Linear(num_filters, 1))

    # Move model to device
    model = model.to(device)

    return model, device


# Get binary cross-entropy loss function.
def get_loss_fn():
    return BCEWithLogitsLoss()


# Calculate the accuracy
def calculate_accuracy(predicted_val, true_val):
    # Calculate accuracy using sigmoid function
    predicted_val = torch.sigmoid(predicted_val)
    predicted_val = (predicted_val > 0.5).float()
    return (predicted_val == true_val).sum() / true_val.size(0)


# Example usage:
""" if __name__ == "__main__":
    model = initialize_model()
    loss_fn = get_loss_fn()
    optimizer = torch.optim.Adam(model.fc.parameters())
    train_step = make_train_step(model, optimizer, loss_fn) """
