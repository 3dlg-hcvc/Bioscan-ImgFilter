import os
import json
import shutil
import argparse
import cv2
from processing_helperFunctions import load_annotations,save_annotations
import random

"""
NOTE:
The cropped images are evaluated for blurriness because
we are interested in the clarity of the insects within the bounding boxes,
not the entire image.This approach ensures that the model trains on and
identifies whole, original images as clear or blurry based on the clarity
of the insects in the bounding boxes.
"""


# Computes Laplacian variance to measure image blurriness
def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()


# Creates directories for storing processed original images.
def create_output_directories(output_dir):

    # create good and bad image directories
    bad_images_dir = os.path.join(output_dir, "bad_imgs")
    good_images_dir = os.path.join(output_dir, "good_imgs")
    blurry_images_dir = os.path.join(output_dir, "blurry_imgs")
    clear_images_dir = os.path.join(output_dir, "clear_imgs")
    cropped_blurry_images_dir = os.path.join(output_dir, "cropped_blurry_imgs")
    cropped_clear_images_dir = os.path.join(output_dir, "cropped_clear_imgs")

    os.makedirs(blurry_images_dir, exist_ok=True)
    os.makedirs(clear_images_dir, exist_ok=True)
    os.makedirs(cropped_blurry_images_dir, exist_ok=True)
    os.makedirs(cropped_clear_images_dir, exist_ok=True)

    return bad_images_dir, good_images_dir, blurry_images_dir, clear_images_dir, cropped_blurry_images_dir, cropped_clear_images_dir

# Randomly select a specified number of images to remain in a directory
def randomly_select_images(directory, num_images_to_keep):
    all_images = os.listdir(directory)
    images_to_keep = random.sample(all_images, num_images_to_keep)
    images_to_remove = set(all_images) - set(images_to_keep)

    for image in images_to_remove:
        os.remove(os.path.join(directory, image))
# Processes and classifies images as blurry or clear based on variance.
def process_images(output_dir, image_mapping, threshold, annotations):
    bad_img_dir, good_img_dir, original_blurry_dir, original_clear_dir, cropped_blurry_images_dir, cropped_clear_images_dir = create_output_directories(output_dir)
    original_clear_annotations = {"images": [], "annotations": []}

    num_original_bad = len(os.listdir(bad_img_dir))

    for cropped_path, original_path in image_mapping.items():
        try:
            image = cv2.imread(cropped_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            focus_measure = variance_of_laplacian(gray)
            original_filename = os.path.basename(original_path)

            if focus_measure < threshold:
                destination_dir = bad_img_dir
                sub_dir = original_blurry_dir
                cropped_dir = cropped_blurry_images_dir


            else:
                destination_dir = good_img_dir
                sub_dir = original_clear_dir
                cropped_dir = cropped_clear_images_dir

                # Find the corresponding image data in annotations
                image_data = next(
                    (
                        img
                        for img in annotations["images"]
                        if img["file_name"] == original_filename
                    ),
                    None,
                )

                # If clear, add image and annotation to clear set
                if image_data:
                    original_clear_annotations["images"].append(image_data)
                    img_annotations = [
                        ann
                        for ann in annotations["annotations"]
                        if ann["image_id"] == image_data["id"]
                    ]
                    original_clear_annotations["annotations"].extend(img_annotations)

            # Check if the image has already been copied
            if not os.path.exists(os.path.join(destination_dir, original_filename)):
                shutil.copy(original_path, os.path.join(destination_dir, original_filename))

            if not os.path.exists(os.path.join(sub_dir, original_filename)):
                shutil.copy(original_path, os.path.join(sub_dir, original_filename))
            
            if not os.path.exists(os.path.join(cropped_dir, original_filename)):
                shutil.copy(cropped_path, os.path.join(cropped_dir, original_filename))


        except Exception as e:
            print(f"Error processing {cropped_path}: {e}")

    # Save the annotations for clear images to a JSON file
    save_annotations(original_clear_annotations, good_img_dir, "original_clear_annotations.json")
    save_annotations(original_clear_annotations, original_clear_dir, "original_clear_annotations.json")

    print(f"Number of blurry images:", len(os.listdir(original_blurry_dir)))
    print(f"Number of clear images:", len(os.listdir(original_clear_dir)))

    print(f"Number of bad images:", len(os.listdir(bad_img_dir)))
    print(f"Number of good images:", len(os.listdir(good_img_dir)))

    print(f"Number of cropped blurry images:", len(os.listdir(cropped_blurry_images_dir)))
    print(f"Number of cropped clear images:", len(os.listdir(cropped_clear_images_dir)))



# load the image mapping file, annotations, and process images as clear or blurry
def main(args):

    # LOad the image mapping 
    image_mapping = load_annotations(args.mapping_dir, "image_mapping.json")

    # Load the annotations
    annotations = load_annotations(args.input_dir, "coco_annotations_processed.json")

    process_images(
        args.output_dir, image_mapping, args.threshold, annotations
    )

    # Randomly select 150 images to remain in the cropped clear images directory
    randomly_select_images(os.path.join(args.output_dir, "cropped_clear_imgs"), 150)
    print(f"Number of images in cropped clear images directory:", len(os.listdir(os.path.join(args.output_dir, "cropped_clear_imgs"))))

  





if __name__ == "__main__":

    # Define file locations
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input_dir",
        required=True,
        help="Path to input directory of cropped images",
    )
    parser.add_argument(
        "-m",
        "--mapping_dir",
        required=True,
        help="Path to JSON mapping file of cropped to original images",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        required=True,
        help="Path to output directory of clear and blurry images",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=130.0,
        help="Focus measures falling below this are considered 'blurry'",
    )
    args = parser.parse_args()

    # Implement blur detection
    main(args)
