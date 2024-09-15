from sklearn.base import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
import torch
import torchvision.transforms as T
from torchvision import datasets
from torch.utils.data import DataLoader
import os
import sys
from shutil import copy2
from glob import glob

# Append the data_processing directory to the system path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../data_processing"))
)
from scripts.dataloaders import get_datasets
from training_helperFunctions import initialize_model


# Directories for processed images
train_val_bad_dir = "./dataset/processed_imgs/bad_imgs"
train_val_good_dir = "./dataset/processed_imgs/good_imgs"
original_dataset_dir = "./dataset/failed_crop_subset"
test_set_path = "./dataset"

test_set_dir = os.path.join(test_set_path,"test_imgs")
os.makedirs(test_set_dir, exist_ok=True)



# Get the filenames used in training and validation sets

train_val_good_bad_arr = []
original_dataset_arr  = []

for img_path in os.listdir(train_val_bad_dir):  # Loop over the actual filenames
    filename = os.path.basename(img_path)
    train_val_good_bad_arr.append(filename)


for img_path in os.listdir(train_val_good_dir):  # Loop over the actual filenames
    filename = os.path.basename(img_path)
    train_val_good_bad_arr.append(filename)


for img_path in os.listdir(original_dataset_dir):  # Loop over the actual filenames
    filename = os.path.basename(img_path)
    original_dataset_arr.append(filename)

print("train and val image length total: ",len(train_val_good_bad_arr))  # Print the 6th filename in the array
print("original image length total: ",len(original_dataset_arr))  # Print the 6th filename in the array

for img_name in original_dataset_arr:
    if img_name not in train_val_good_bad_arr:
        copy2(os.path.join(original_dataset_dir, img_name), test_set_dir)

print(len(os.listdir(test_set_dir)))

# # Initialize model
model, device = initialize_model()


# Load test dataset
test_transform = T.Compose([
    T.Resize((224, 224)),  # Resize to the input size expected by the model
    T.ToTensor(),
])
test_dataset = datasets.ImageFolder(test_set_dir, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Evaluate the model on the entire test set
def eval_test_set(test_loader, model, device):
    model.eval()  # Set the model to evaluation mode

    # Ensure no gradients are calculated during inference
    with torch.no_grad():
        for i, (sample, _) in enumerate(test_loader):
            sample = sample.to(device)

            prediction = torch.sigmoid(model(sample))
            prediction_value = prediction.item() * 100

            if prediction < 0.5:
                print(f"Image {i+1}: Bad Image")
                print(f"Percent good: {prediction_value:.2f}%")
            else:
                print(f"Image {i+1}: Good Image")
                print(f"Percent good: {prediction_value:.2f}%")

            # Transform tensor to PIL image
            transform = T.ToPILImage()
            img = transform(
                sample.squeeze(0).cpu().clamp(0, 1)
            )  # Remove the batch dimension, move to CPU, and clamp values to [0, 1]

        
            # # Accuracy, F1 Score, and los over the entire epoch
            # val_f1_score = f1_score(all_val_targets, all_val_predictions)
            # val_accuracy = accuracy_score(all_val_targets, all_val_predictions)
            # precision = precision_score(all_val_targets, all_val_predictions)
            # recall = recall_score(all_val_targets, all_val_predictions)
            #     # Compute confusion matrix

            # conf_matrix = confusion_matrix(all_val_targets, all_val_predictions)
            # class_report = classification_report(all_val_targets, all_val_predictions, target_names=['Blurry', 'Clear'])

            # print(
            #     f"Epoch: {epoch+1}, Val Loss: {total_val_loss:.4f}, Val Accuracy: {val_accuracy*100:.4f}%, "
            #     f"Val F1 Score: {val_f1_score:.4f}, Val Precision Score: {precision:.4f}, Val Recall Score: {recall:.4f}, "
            #     f"\nConfusion matrix:\n",conf_matrix, 
            #     f"\n",class_report
            # )

            #img.show()

# Calling the function to evaluate the entire test set
#eval_test_set(test_loader, model, device)
