import os
import argparse
import random
import matplotlib.pyplot as plt
from PIL import Image
import torch
from tqdm import tqdm
from dataloaders import get_data_loaders, get_datasets
from trainingFunctions import initialize_model, get_loss_fn, make_train_step
from training import train_model,calculate_accuracy

import matplotlib.pyplot as plt


# displays sample good and bad images 
def display_sample_images(folder, num_samples=3):
    label = os.path.basename(folder)
    images = os.listdir(folder)
    sample_images = random.sample(images, min(num_samples, len(images)))
    
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 5))
    fig.suptitle(f'Sample Images: {label}', fontsize=16)
    
    for ax, img_name in zip(axes, sample_images):
        img_path = os.path.join(folder, img_name)
        img = Image.open(img_path)
        ax.imshow(img)
        ax.axis('off')

    plt.show()



# Plots the training and validation loss and accuracy values over each epoch
def plot_metrics(train_values, val_values, metric_name, ylabel, title, save_filename):
    epochs = range(1, len(train_values) + 1)  # Number of epochs

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_values, 'o-', label=f'Training {metric_name}', color='blue')
    plt.plot(epochs, val_values, 's-', label=f'Validation {metric_name}', color='red')
    
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.xticks(epochs)  # Set x-axis ticks to show every epoch
    plt.savefig(save_filename)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--goodTrain_folder_path', type=str, required=True, help="Path to the 'good' training folder")
    parser.add_argument('--badTrain_folder_path', type=str, required=True, help="Path to the 'bad' training folder")
    args = parser.parse_args()

    #print("Displaying sample images from 'good' training set:")
    #display_sample_images(args.goodTrain_folder_path)

    #print("Displaying sample images from 'bad' training set:")
    #display_sample_images(args.badTrain_folder_path)


    # Define the directories containing the data
    train_dir = "./data/data_splits/train"
    val_dir = "./data/data_splits/val"

    # Call the function to get the datasets
    train_data, val_data = get_datasets(train_dir, val_dir)

    # Specify the batch size for the data loaders
    batch_size = 4

    # Call the function to get the data loaders
    train_loader, val_loader = get_data_loaders(train_data, val_data, batch_size)

    # Train the model
    # Train the model and get the loss and accuracy history
    trained_model, train_loss, val_loss, train_acc, val_acc = train_model(train_loader, val_loader)

    # Plotting loss
    plot_metrics(train_loss, val_loss, "Loss", "Loss", "Training and Validation Loss over Epochs", "Train_and_Val_Loss.png")

    # Plotting accuracy
    plot_metrics(train_acc, val_acc, "Accuracy", "Accuracy (%)", "Training and Validation Accuracy over Epochs", "Train_and_Val_Accuracy.png")
