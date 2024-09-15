from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
import torch
import sys
import os
import wandb
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse
from training_helperFunctions import misclassified_imgs



from training_helperFunctions import (
    initialize_model,
    get_loss_fn,
    train_step
)

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../data_processing"))
)
# from scripts.dataloaders import get_data_loaders, get_datasets
from dataloaders import get_data_loaders, get_datasets


def train_model(train_loader, val_loader, n_epochs=60, use_wandb=False):
    # Initialize the model,loss, optimizer,and training step function
    model, device = initialize_model()
    loss_fn = get_loss_fn()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-6)
    scheduler = ReduceLROnPlateau(optimizer, "min", patience=2, factor=0.1)

    epoch_train_loss = []
    epoch_val_loss = []
    epoch_train_acc = []
    epoch_val_acc = []

    # Initialize variables for tracking the best model weights and loss
    best_model_wts = model.state_dict()
    best_loss = float("inf")
    
    if use_wandb:
        wandb.init(
            # track hyperparameters and run metadata
            config={
                "learning_rate": 0.001
            },
        )

    # Loop through each epoch
    for epoch in range(n_epochs):
        epoch_loss = 0
        all_train_targets = []
        all_train_predictions = []

        # Iterate through the training dataset
        for i, data in tqdm(enumerate(train_loader), total=len(train_loader)):
            input_batch, target_labels_batch = data
            input_batch = input_batch.to(device)
            target_labels_batch = target_labels_batch.unsqueeze(1).float().to(device)

            # Calculate loss and update model parameters
            loss = train_step(
                model, optimizer, loss_fn, input_batch, target_labels_batch
            )
            epoch_loss += loss.item() / len(train_loader)

            # Calculate training accuracy
            input_predictions = model(input_batch)

            # Calculating good/bad class
            predicted_labels = torch.sigmoid(input_predictions) > 0.5
            all_train_targets.extend(target_labels_batch.cpu().numpy())
            all_train_predictions.extend(predicted_labels.cpu().numpy())


        train_accuracy = accuracy_score(all_train_targets,all_train_predictions)
        train_f1_score = f1_score(all_train_targets, all_train_predictions)

        print(
            f"\nEpoch: {epoch+1}, Train Loss: {epoch_loss:.4f}, Train Accuracy: {train_accuracy*100:.4f}%, Train F1 Score: {train_f1_score:.4f}"
        )
        epoch_train_acc.append(train_accuracy * 100)
        epoch_train_loss.append(epoch_loss)

        # Set the model to evaluation mode
        model.eval()
        with torch.no_grad():
            total_val_loss = 0
            all_val_predictions = []
            all_val_targets = []

            # Iterate through the validation dataset
            for input_batch, target_labels_batch in val_loader:
                input_batch = input_batch.to(device)
                target_labels_batch = (
                    target_labels_batch.unsqueeze(1).float().to(device)
                )

                # Calculate validation loss
                input_predictions = model(input_batch)
                val_loss = loss_fn(input_predictions, target_labels_batch)
                total_val_loss += val_loss.item() / len(val_loader)

                # Calculate vallidation F1 Score
                predicted_labels = torch.sigmoid(input_predictions) > 0.5
                all_val_targets.extend(target_labels_batch.cpu().numpy())
                all_val_predictions.extend(predicted_labels.cpu().numpy())

                # Save misclassified images during the last epoch
                if epoch == n_epochs - 1:
                    misclassified_imgs(input_batch, predicted_labels, target_labels_batch)


            # Accuracy, F1 Score, and los over the entire epoch
            val_f1_score = f1_score(all_val_targets, all_val_predictions)
            val_accuracy = accuracy_score(all_val_targets, all_val_predictions)
            precision = precision_score(all_val_targets, all_val_predictions)
            recall = recall_score(all_val_targets, all_val_predictions)
            
            # Compute confusion matrix
            conf_matrix = confusion_matrix(all_val_targets, all_val_predictions)
            class_report = classification_report(all_val_targets, all_val_predictions, target_names=['Empty', 'Blurry'])

            print(
                f"Epoch: {epoch+1}, Val Loss: {total_val_loss:.4f}, Val Accuracy: {val_accuracy*100:.4f}%, "
                f"Val F1 Score: {val_f1_score:.4f}, Val Precision Score: {precision:.4f}, Val Recall Score: {recall:.4f}, "
                f"\nConfusion matrix:\n",conf_matrix, 
                f"\n",class_report
            )

            epoch_val_loss.append(total_val_loss)
            epoch_val_acc.append(val_accuracy * 100)

            if use_wandb:
                wandb.log(
                    {
                        "epoch": epoch + 1,
                        "train_loss": epoch_loss,
                        "train_accuracy": train_accuracy,
                        "train_f1_score": train_f1_score,
                        "val_loss": total_val_loss,
                        "val_accuracy": val_accuracy,
                        "val_f1_score": val_f1_score,
                        "val_precision": precision,
                        "val_recall": recall,
                    }
                )

            best_loss = min(epoch_val_loss)

            # Update best model weights if validation loss improves
            if total_val_loss < best_loss:
                best_model_wts = model.state_dict()

        scheduler.step(total_val_loss)

    # Load the best model weights
    model.load_state_dict(best_model_wts)


    # Finish the wandb run
    if use_wandb:
        wandb.finish()
    return model, epoch_train_loss, epoch_val_loss, epoch_train_acc, epoch_val_acc


def main():

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train a model with optional wandb logging")
    parser.add_argument("--use_wandb", type=bool, default=False, help="Enable wandb logging")
    args = parser.parse_args()
    
    # Define the directories containing the data
    train_dir = "./dataset/data_splits/train"
    val_dir = "./dataset/data_splits/val"

    # Call the function to get the datasets
    train_data, val_data = get_datasets(train_dir, val_dir)

    # Specify the batch size for the data loaders
    batch_size = 10

    # Call the function to get the data loaders
    train_loader, val_loader = get_data_loaders(train_data, val_data, batch_size)

    # Train the model and get the loss history
    train_model(train_loader, val_loader, use_wandb=args.use_wandb)


if __name__ == "__main__":
    main()
