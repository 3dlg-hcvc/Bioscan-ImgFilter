from torch.nn.modules.loss import BCEWithLogitsLoss
from torch.optim import lr_scheduler
import torch
import torchvision.models as models
from torchvision.models import ResNet18_Weights
import torch.nn as nn


# Training step for neural network 
def make_train_step(model, optimizer, loss_fn):
  def train_step(input_data,target_labels):
    #make prediction
    input_predictions = model(input_data)
    #enter train mode
    model.train()
    
    #compute loss
    loss = loss_fn(input_predictions,target_labels)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    #optimizer.cleargrads()

    return loss
  return train_step


 
# Initialize a ResNet18 model with a custom final layer.
def initialize_model():
   
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    # Add a new final layer
    num_filters = model.fc.in_features
    model.fc = nn.Linear(num_filters, 1)

    # Move model to device
    model = model.to(device)

    return model, device


# Get binary cross-entropy loss function.
def get_loss_fn():
    return BCEWithLogitsLoss()


# Example usage:
""" if __name__ == "__main__":
    model = initialize_model()
    loss_fn = get_loss_fn()
    optimizer = torch.optim.Adam(model.fc.parameters())
    train_step = make_train_step(model, optimizer, loss_fn) """