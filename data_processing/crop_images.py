import os
import shutil
from PIL import Image
from processing_helperFunctions import load_annotations, save_annotations, create_output_directories


# Crop image, store in respective directories, map cropped image to original
def process_crop_image(image_path, img_annotations, cropped_img_dir, empty_img_dir, image_mapping):

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
                cropped_img_dir, cropped_image_filename
            )
            cropped_image.save(cropped_image_path)

            # Save the cropped image path to the original
            image_mapping[cropped_image_path] = image_path

    # If the image is unbounded, move it to the unbounded directory
    if is_unbounded:
        shutil.copy(
            image_path, os.path.join(empty_img_dir, os.path.basename(image_path))
        )


# Crop the image, store image in respective directories, create image mapping json file
def process_image(input_dir,output_dir,directories):

    # Load the image annotations
    coco_annotation_dict = load_annotations(input_dir, "coco_annotations_processed.json")
    images, annotations = (
        coco_annotation_dict["images"],
        coco_annotation_dict["annotations"],
    )


    # Create directories for image storage
    cropped_img_dir, empty_img_dir = create_output_directories(output_dir,directories)
    

    # Initialize image path mapping dictionary
    image_mapping = {}
    for img in images:
        image_id = img["id"]
        image_path = os.path.join(input_dir, img["file_name"])
        # Ensure that each image has a corresponding annotation
        img_annotations = [ann for ann in annotations if ann["image_id"] == image_id]

        # Crop the image and map the cropped path to the original
        process_crop_image(
            image_path,
            img_annotations,
            cropped_img_dir,
            empty_img_dir,
            image_mapping
        )

    # Save the image mappings into a json file
    save_annotations(image_mapping, cropped_img_dir, "image_mapping.json")

    num_unbounded = len(os.listdir(empty_img_dir))
    num_cropped = len(os.listdir(cropped_img_dir))

    print(num_cropped, f"Cropped images saved in {cropped_img_dir}")
    print(num_unbounded, f"Unbounded images saved in {empty_img_dir}")


if __name__ == "__main__":

    input_dir = "dataset/failed_crop_subset"
    output_dir = "dataset/filtered_imgs/invalid_imgs"

    directories= ["cropped_imgs","empty_imgs"]

    process_image(input_dir,output_dir,directories)