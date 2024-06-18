from sklearn.metrics import f1_score
import torch
from tqdm import tqdm
from dataloaders import get_data_loaders, get_datasets
from trainingFunctions import initialize_model, get_loss_fn, train_step



def train_model(train_loader, val_loader, n_epochs=10):
    # Initialize the model, loss function, optimizer, and training step function
    model, device = initialize_model()
    loss_fn = get_loss_fn()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)

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
        all_train_targets = []
        all_train_predictions = []

        # Set the model to training mode
        model.train()

        # Iterate through the training dataset
        for i, data in tqdm(enumerate(train_loader), total=len(train_loader)):
            input_batch, target_labels_batch = data
            input_batch = input_batch.to(device)
            target_labels_batch = target_labels_batch.unsqueeze(1).float().to(device)

            # Calculate loss and update model parameters
            loss = train_step(model, optimizer, loss_fn,input_batch, target_labels_batch)
            epoch_loss += loss.item() / len(train_loader)
            
            # Calculate training accuracy
            input_predictions = model(input_batch)
            correct_predictions += calculate_accuracy(input_predictions, target_labels_batch)

            # Calculating F1 Score 
            predicted_labels = torch.sigmoid(input_predictions) > 0.5
            all_train_targets.extend(target_labels_batch.cpu().numpy())
            all_train_predictions.extend(predicted_labels.cpu().numpy())


        train_accuracy = correct_predictions.item() / len(train_loader)
        train_f1_score = f1_score(all_train_targets, all_train_predictions)

        
        print(f'\nEpoch: {epoch+1}, Train Loss: {epoch_loss:.4f}, Train Accuracy: {train_accuracy*100:.4f}%, Train F1 Score: {train_f1_score:.4f}')
        epoch_train_acc.append(train_accuracy*100)        
        epoch_train_loss.append(epoch_loss)

        # Set the model to evaluation mode
        model.eval()
        with torch.no_grad():
            total_val_loss = 0
            correct_predictions = 0
            all_val_predictions = []
            all_val_targets = []
            

            # Iterate through the validation dataset
            for input_batch, target_labels_batch in val_loader:
                input_batch = input_batch.to(device)
                target_labels_batch = target_labels_batch.unsqueeze(1).float().to(device)
                

                # Calculate validation loss
                input_predictions = model(input_batch)
                val_loss = loss_fn(input_predictions, target_labels_batch)
                total_val_loss += val_loss.item()/len(val_loader)
                
                
                # Calculate validation accuracy
                correct_predictions += calculate_accuracy(input_predictions, target_labels_batch)

                # Calculate vallidation F1 Score 
                predicted_labels = torch.sigmoid(input_predictions) > 0.5
                all_val_targets.extend(target_labels_batch.cpu().numpy())
                all_val_predictions.extend(predicted_labels.cpu().numpy())

            # Accuracy, F1 Score, and los over the entire epoch 
            val_accuracy = correct_predictions.item() / len(val_loader)
            val_f1_score = f1_score(all_val_targets,all_val_predictions)


            print(f'Epoch: {epoch+1}, Val Loss: {total_val_loss:4f}, Val Accuracy: {val_accuracy*100:.4f}%, Val F1 Score: {val_f1_score:.4f}')
            epoch_val_loss.append(total_val_loss)
            epoch_val_acc.append(val_accuracy*100)

            best_loss = min(epoch_val_loss)

            # Update best model weights if validation loss improves
            if total_val_loss < best_loss:
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



def main():
    # Define the directories containing the data
    train_dir = "./data/data_splits/train"
    val_dir = "./data/data_splits/val"

    # Call the function to get the datasets
    train_data, val_data = get_datasets(train_dir, val_dir)

    # Specify the batch size for the data loaders
    batch_size = 32

    # Call the function to get the data loaders
    train_loader, val_loader = get_data_loaders(train_data, val_data, batch_size)

    # Train the model and get the loss history
    trained_model, train_loss, val_loss, train_acc, val_acc = train_model(train_loader, val_loader)

    

if __name__ == "__main__":
    main()