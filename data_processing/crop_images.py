import os
import json
import shutil
from PIL import Image
import argparse

# Load all image annotations 
def load_annotations(input_dir):
    with open(os.path.join(input_dir, "coco_annotations_processed.json")) as f:
        return json.load(f)

# Create directories to store cropped and unbounded images 
def create_directories(cropped_output_dir, unbounded_output_dir):
    os.makedirs(cropped_output_dir, exist_ok=True)
    os.makedirs(unbounded_output_dir, exist_ok=True)

# Crop the image, store in respective directories, map cropped image path to original 
def crop_image(image_path, img_annotations, cropped_output_dir, unbounded_output_dir, image_mapping):

    # Check if the image path is valid
    if not os.path.exists(image_path):
        #print(f"Image {os.path.basename(image_path)} not found.")
        print(image_path)
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
            cropped_image_path = os.path.join(cropped_output_dir, cropped_image_filename)
            cropped_image.save(cropped_image_path)

            # Save the cropped image path to the original 
            image_mapping[cropped_image_path] = image_path

    # If the image is unbounded, move it to the unbounded directory
    if is_unbounded:
        shutil.copy(image_path, os.path.join(unbounded_output_dir, os.path.basename(image_path)))

# Save the cropped to original path mappings in a json file 
def save_image_mapping(cropped_output_dir, image_mapping):
    with open(os.path.join(cropped_output_dir, "image_mapping.json"), "w") as f:
        json.dump(image_mapping, f, indent=4)

# Crop the image, store image in respective directories, create image mapping json file
def process_image(args):

    # Load the image annotations 
    coco_annotation_dict = load_annotations(args.input_dir)
    images, annotations = coco_annotation_dict["images"], coco_annotation_dict["annotations"]

    # Create cropped and unbounded directories for image storage 
    create_directories(args.cropped_output_dir, args.unbounded_output_dir)

    # Initialize image path mapping dictionary 
    image_mapping = {}
    for img in images:
        image_id = img["id"]
        image_path = os.path.join(args.input_dir, img["file_name"])
        # Ensure that each image has a corresponding annotation 
        img_annotations = [ann for ann in annotations if ann["image_id"] == image_id]
        
        # Crop the image and map the cropped path to the original 
        crop_image(image_path, img_annotations, args.cropped_output_dir, args.unbounded_output_dir, image_mapping)
    
    # Save the image mappings into a json file 
    save_image_mapping(args.cropped_output_dir, image_mapping)

    num_unbounded = len(os.listdir(args.unbounded_output_dir))
    num_cropped = len(os.listdir(args.cropped_output_dir))

    
    print(num_cropped, f"Cropped images saved in {args.cropped_output_dir}")
    print(num_unbounded, f"Unbounded images saved in {args.unbounded_output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Path to the folder that contains the images and COCO file.")
    parser.add_argument("--cropped_output_dir", type=str, required=True, help="Path to save cropped images.")
    parser.add_argument("--unbounded_output_dir", type=str, required=True, help="Path to save unbounded images.")
    args = parser.parse_args()

    process_image(args)

