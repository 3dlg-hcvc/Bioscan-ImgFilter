from sklearn import metrics
import torch
from tqdm import tqdm
from dataloaders import get_data_loaders, get_datasets
from trainingFunctions import initialize_model, get_loss_fn, make_train_step
import matplotlib.pyplot as plt


def train_model(train_loader, val_loader, n_epochs=10):
    # Initialize the model, loss function, optimizer, and training step function
    model, device = initialize_model()
    loss_fn = get_loss_fn()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)
    train_step = make_train_step(model, optimizer, loss_fn)

    epoch_train_loss = []
    epoch_val_loss = []
    epoch_train_acc = []
    epoch_val_acc = []

    # Initialize variables for tracking the best model weights and loss
    best_model_wts = model.state_dict()
    best_loss = float('inf')

    # Loop through each epoch
    for epoch in range(n_epochs):
        epoch_loss = 0
        correct_predictions = 0
        total_predictions = 0

        # Set the model to training mode
        model.train()

        # Iterate through the training dataset
        for i, data in tqdm(enumerate(train_loader), total=len(train_loader)):
            input_batch, target_labels_batch = data
            input_batch = input_batch.to(device)
            target_labels_batch = target_labels_batch.unsqueeze(1).float().to(device)

            # Calculate loss and update model parameters
            loss = train_step(input_batch, target_labels_batch)
            epoch_loss += loss.item() / len(train_loader)
            
            # Calculate training accuracy
            with torch.no_grad():
                input_predictions = model(input_batch)
                correct_predictions += calculate_accuracy(input_predictions, target_labels_batch)

        train_accuracy = correct_predictions.item() / len(train_loader)
        
        print(f'\nEpoch: {epoch+1}, Train Loss: {epoch_loss}, Train Accuracy: {train_accuracy*100:.4f}%')
        epoch_train_acc.append(train_accuracy*100)
        
        epoch_train_loss.append(epoch_loss)
        # Set the model to evaluation mode
        model.eval()
        with torch.no_grad():
            cum_loss = 0
            correct_predictions = 0
            total_predictions = 0

            # Iterate through the validation dataset
            for input_batch, target_labels_batch in val_loader:
                input_batch = input_batch.to(device)
                target_labels_batch = target_labels_batch.unsqueeze(1).float().to(device)

                # Calculate validation loss
                input_predictions = model(input_batch)
                val_loss = loss_fn(input_predictions, target_labels_batch)
                cum_loss += val_loss.item() / len(val_loader)
                
                
                # Calculate validation accuracy
                correct_predictions += calculate_accuracy(input_predictions, target_labels_batch)

            val_accuracy = correct_predictions.item() / len(val_loader)
        
            print(f'Epoch: {epoch+1}, Val Loss: {cum_loss}, Val Accuracy: {val_accuracy*100:.4f}%')
            epoch_val_loss.append(cum_loss)
            epoch_val_acc.append(val_accuracy*100)

            # Update best model weights if validation loss improves
            if cum_loss < best_loss:
                best_loss = cum_loss
                best_model_wts = model.state_dict()

    # Load the best model weights
    model.load_state_dict(best_model_wts)
    #print(epoch_val_acc,epoch_train_acc)
    return model, epoch_train_loss, epoch_val_loss, epoch_train_acc, epoch_val_acc




def calculate_accuracy(predicted_val, true_val):
    # Calculate accuracy using sigmoid function
    predicted_val = torch.sigmoid(predicted_val)
    predicted_val = (predicted_val > 0.5).float()
    return (predicted_val == true_val).sum()/true_val.size(0)


def get_accuracy(y_true, y_prob):
    assert y_true.ndim == 1 and y_true.size() == y_prob.size()
    y_prob = y_prob > 0.5
    return (y_true == y_prob).sum().item() / y_true.size(0)


def main():
    # Define the directories containing the data
    train_dir = "./data/data_splits/train"
    val_dir = "./data/data_splits/val"

    # Call the function to get the datasets
    train_data, val_data = get_datasets(train_dir, val_dir)

    # Specify the batch size for the data loaders
    batch_size = 4

    # Call the function to get the data loaders
    train_loader, val_loader = get_data_loaders(train_data, val_data, batch_size)

    # Train the model and get the loss history
    trained_model, train_loss, val_loss, train_acc, val_acc = train_model(train_loader, val_loader)

    

if __name__ == "__main__":
    main()