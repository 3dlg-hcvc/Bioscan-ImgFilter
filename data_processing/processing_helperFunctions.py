import os
import json
import random
import shutil
from torchvision import transforms
from PIL import Image
import random

# Load image annotations 
def load_annotations(input_dir, file_name):
    with open(os.path.join(input_dir, file_name)) as f:
        return json.load(f)
    

# Save updated annotations back to JSON file.
def save_annotations(annotations,output_dir, file_name):
    with open(os.path.join(output_dir, file_name), "w") as f:
        json.dump(annotations, f, indent=4)


# deletes a directory 
def delete_directory(path):
    shutil.rmtree(path) 


def move_directory(source, dest):
    shutil.move(source, dest)


# Creates directories for storing image classes
def create_output_directories(output_dir,directories):

    updated_arr = []
    #updated_map = {}

    for dir_name in directories:
        # Construct the full path
        full_path = os.path.join(output_dir, dir_name)
        
        # Create the directory
        os.makedirs(full_path, exist_ok=True)
        
        # Assign the created directory path to the variable
        updated_arr.append(full_path)
    
    # return all directory paths
    return updated_arr
    


def randomly_select_copy_images(source_directory, destination_directory, num_images_to_select):

    # Create the destination directory if it doesn't exist
    if not os.path.exists(destination_directory):
        os.makedirs(destination_directory)

    # List all images in the source directory
    all_images = os.listdir(source_directory)
    current_images = os.listdir(destination_directory)

    if len(current_images)>=num_images_to_select:
        return 

    # Randomly select the specified number of images
    selected_images = random.sample(all_images, num_images_to_select)

    # Copy selected images to the destination directory
    for image in selected_images:
        source_path = os.path.join(source_directory, image)
        destination_path = os.path.join(destination_directory, image)
        # Only copy if the image doesn't already exist in the destination directory
        if not os.path.exists(destination_path):
            shutil.copy2(source_path, destination_path)



def augment_image(image_path, output_dir, num_augmentations=2):
    """Apply a specified number of augmentations to an image and save the augmented images."""
    # Load the image
    image = Image.open(image_path)
    
    # Define possible augmentations
    augmentations = [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.Resize((224, 224)),
        transforms.Pad(padding = [1,2,3,6], fill=3, padding_mode='constant'),
        transforms.Grayscale(num_output_channels=1)
    ]
    
    base_name = os.path.basename(image_path)
    name, ext = os.path.splitext(base_name)
    
    for i in range(num_augmentations):
        # Randomly select a subset of augmentations to apply
        selected_transforms = random.sample(augmentations, k=random.randint(1, len(augmentations)))
        transform_chain = transforms.Compose(selected_transforms)
        
        # Convert PIL image to tensor, apply transformations, and convert back to PIL image
        image_tensor = transforms.ToTensor()(image)
        augmented_image_tensor = transform_chain(image_tensor)
        augmented_image = transforms.ToPILImage()(augmented_image_tensor)
        
        # Save the augmented image
        augmented_image_path = os.path.join(output_dir, f"{name}_aug_{i+1}{ext}")
        augmented_image.save(augmented_image_path)



