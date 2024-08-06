import os
import json
import shutil
from PIL import Image
import argparse
from processing_helperFunctions import load_annotations, save_annotations


# Create directories to store cropped and unbounded images
def create_directories(cropped_output_dir, unbounded_output_dir, unbounded_labelled, bounded_labelled):
    os.makedirs(cropped_output_dir, exist_ok=True)
    os.makedirs(unbounded_output_dir, exist_ok=True)

    os.makedirs(unbounded_labelled, exist_ok=True)
    os.makedirs(bounded_labelled, exist_ok=True)


# Creates directories for storing processed original images.
def create_output_directories(output_dir):

    # create good and bad image directories
    bad_images_dir = os.path.join(output_dir, "bad_imgs")
    good_images_dir = os.path.join(output_dir, "good_imgs")
    cropped_output_dir = os.path.join(output_dir,"cropped_imgs")
    unbounded_output_dir = os.path.join(output_dir,"unbounded_imgs")


    os.makedirs(cropped_output_dir, exist_ok=True)
    os.makedirs(unbounded_output_dir, exist_ok=True)
    os.makedirs(good_images_dir, exist_ok=True)
    os.makedirs(bad_images_dir, exist_ok=True)
    return bad_images_dir, good_images_dir, cropped_output_dir, unbounded_output_dir


# Crop image, store in respective directories, map cropped image to original
def crop_image(image_path, img_annotations, cropped_output_dir, unbounded_output_dir, bad_images_dir, image_mapping):

    # Check if the image path is valid
    if not os.path.exists(image_path):
        print(f"Image {os.path.basename(image_path)} not found.")
        return

    # Open the image
    image = Image.open(image_path)

    # Flag to check if the image is unbounded
    is_unbounded = True

    # Loop through each image's annotations
    for ann in img_annotations:
        bbox = ann["bbox"]
        left, top, width, height = bbox

        # Move images to the unbounded directory if it does not have a valid bounding box
        if width > 0 and height > 0:
            is_unbounded = False
            right, bottom = left + width, top + height
            cropped_image = image.crop((left, top, right, bottom))

            # Create the cropped image filename, move to the cropped directory, and save its path
            cropped_image_filename = f"{ann['image_id']}_{ann['id']}_cropped.jpg"
            cropped_image_path = os.path.join(
                cropped_output_dir, cropped_image_filename
            )
            cropped_image.save(cropped_image_path)

            # Save the cropped image path to the original
            image_mapping[cropped_image_path] = image_path

    # If the image is unbounded, move it to the unbounded directory
    if is_unbounded:
        shutil.copy(
            image_path, os.path.join(unbounded_output_dir, os.path.basename(image_path))
        )
        shutil.copy(
            image_path, os.path.join(bad_images_dir, os.path.basename(image_path))
        )


# Crop the image, store image in respective directories, create image mapping json file
def process_image(args):

    # Load the image annotations
    coco_annotation_dict = load_annotations(args.input_dir, "coco_annotations_processed.json")
    images, annotations = (
        coco_annotation_dict["images"],
        coco_annotation_dict["annotations"],
    )

    # Create cropped and unbounded directories for image storage
    #create_directories(args.output_dir)

    # Create directories for image storage
    bad_images_dir, good_images_dir, cropped_output_dir, unbounded_output_dir = create_output_directories(args.output_dir)
    

    # Initialize image path mapping dictionary
    image_mapping = {}
    for img in images:
        image_id = img["id"]
        image_path = os.path.join(args.input_dir, img["file_name"])
        # Ensure that each image has a corresponding annotation
        img_annotations = [ann for ann in annotations if ann["image_id"] == image_id]

        # Crop the image and map the cropped path to the original
        crop_image(
            image_path,
            img_annotations,
            cropped_output_dir,
            unbounded_output_dir,
            bad_images_dir,
            image_mapping,
        )

    # Save the image mappings into a json file
    save_annotations(image_mapping, cropped_output_dir, "image_mapping.json")

    num_unbounded = len(os.listdir(unbounded_output_dir))
    num_cropped = len(os.listdir(cropped_output_dir))

    print(num_cropped, f"Cropped images saved in {cropped_output_dir}")
    print(num_unbounded, f"Unbounded images saved in {unbounded_output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Path to the folder that contains the images and COCO file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save processed images and output directories.")

    args = parser.parse_args()

    process_image(args)