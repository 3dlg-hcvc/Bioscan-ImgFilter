import os
import shutil
import argparse
from sklearn.model_selection import train_test_split
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data_processing')))
from processing_helperFunctions import create_output_directories

#from initial_filtering import augment_img_dir
#from processing_helperFunctions import augment_image



def split_data(input_dir, output_dir ):
    # Create output directories
    init_directories = ["train","val","test"]
    train_test_val_directories = ["uncropped_blurry_imgs","empty_imgs"]

    train_path, val_path, test_path = create_output_directories(output_dir,init_directories)

    uncropped_blurry_train_folder_path, empty_train_folder_path = create_output_directories(train_path,train_test_val_directories)
    uncropped_blurry_val_folder_path,empty_val_folder_path = create_output_directories(val_path,train_test_val_directories)
    uncropped_blurry_test_folder_path,empty_test_folder_path = create_output_directories(test_path,train_test_val_directories)


    # Gather all image file names from uncropped_blurry and empty directories
    uncropped_blurry_images = [
        f
        for f in os.listdir(os.path.join(input_dir, "uncropped_blurry_imgs"))
        if f.endswith(".jpg")
    ]
    empty_images = [
        f
        for f in os.listdir(os.path.join(input_dir, "empty_imgs"))
        if f.endswith(".jpg")
    ]


    # Combine uncropped_blurry and empty images into a single list
    all_images = uncropped_blurry_images + empty_images
    # Create labels for stratification (1 for uncropped_blurry, 0 for empty)
    labels = [1] * len(uncropped_blurry_images) + [0] * len(empty_images)

    # Stratified split for train, validation, and test sets (70% train, 15% val, 15% test)
    train_imgs, temp_imgs, train_labels, temp_labels = train_test_split(
        all_images, labels, test_size=0.3, stratify=labels, random_state=42
    )

    val_imgs, test_imgs, val_labels, test_labels = train_test_split(
        temp_imgs, temp_labels, test_size=0.5, stratify=temp_labels, random_state=42
    )



    # Copy images to the respective directories

    for img, label in zip(train_imgs, train_labels):
        if label == 1:  # uncropped_blurry
            shutil.copyfile(os.path.join(input_dir, "uncropped_blurry_imgs", img), os.path.join(uncropped_blurry_train_folder_path, img))
        else:  # empty
            shutil.copyfile(os.path.join(input_dir, "empty_imgs", img), os.path.join(empty_train_folder_path, img))

    for img, label in zip(val_imgs, val_labels):
        if label == 1:  # uncropped_blurry
            shutil.copyfile(os.path.join(input_dir, "uncropped_blurry_imgs", img), os.path.join(uncropped_blurry_val_folder_path, img))
        else:  # empty
            shutil.copyfile(os.path.join(input_dir, "empty_imgs", img), os.path.join(empty_val_folder_path, img))

    for img, label in zip(test_imgs, test_labels):
        if label == 1:  # uncropped_blurry
            shutil.copyfile(os.path.join(input_dir, "uncropped_blurry_imgs", img), os.path.join(uncropped_blurry_test_folder_path, img))
        else:  # empty
            shutil.copyfile(os.path.join(input_dir, "empty_imgs", img), os.path.join(empty_test_folder_path, img))


    return (train_imgs, test_imgs, val_imgs, 
            uncropped_blurry_train_folder_path, uncropped_blurry_test_folder_path, uncropped_blurry_val_folder_path, 
            empty_train_folder_path,empty_test_folder_path, empty_val_folder_path)
 


def count_dir_imgs(directory_path):
    imgs = os.listdir(directory_path)
    count = 0
    for img in imgs:
        if os.path.isfile(os.path.join(directory_path, img)):
            count += 1
    return count

    
    
def main():
    # Define the directories containing the data
    input_dir = "dataset/filtered_imgs/invalid_imgs" 
    output_dir = "dataset/data_splits"

    # Get paths for different data splits
    (train_imgs, test_imgs, val_imgs,
     uncropped_blurry_train_folder_path, uncropped_blurry_val_folder_path, 
     uncropped_blurry_test_folder_path, empty_train_folder_path, 
     empty_val_folder_path, empty_test_folder_path) = split_data(input_dir, output_dir)

    # Print the number of images in each split
    print(f"\nNumber of Train Images: {len(train_imgs)}")
    print(f"Number of Validation Images: {len(val_imgs)}")
    print(f"Number of Test Images: {len(test_imgs)}\n")

    # Count and print images in each directory
    print(f"uncropped_blurry Train Images: {count_dir_imgs(uncropped_blurry_train_folder_path)}")
    print(f"uncropped_blurry Validation Images: {count_dir_imgs(uncropped_blurry_val_folder_path)}")
    print(f"uncropped_blurry Test Images: {count_dir_imgs(uncropped_blurry_test_folder_path)}\n")
    print(f"Empty Train Images: {count_dir_imgs(empty_train_folder_path)}")
    print(f"Empty Validation Images: {count_dir_imgs(empty_val_folder_path)}")
    print(f"Empty Test Images: {count_dir_imgs(empty_test_folder_path)}\n")


if __name__ == "__main__":
    main()
