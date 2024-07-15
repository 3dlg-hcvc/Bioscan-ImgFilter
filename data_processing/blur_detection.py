import os
import json
import shutil
import argparse
import cv2
from processing_helperFunctions import load_annotations,save_annotations

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

    os.makedirs(good_images_dir, exist_ok=True)
    return bad_images_dir, good_images_dir


# Processes and classifies images as blurry or clear based on variance.
def process_images(output_dir, image_mapping, threshold, annotations):
    original_blurry_dir, original_clear_dir = create_output_directories(output_dir)
    original_clear_annotations = {"images": [], "annotations": []}

    num_original_bad = len(os.listdir(original_blurry_dir))

    for cropped_path, original_path in image_mapping.items():
        try:
            image = cv2.imread(cropped_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            focus_measure = variance_of_laplacian(gray)
            original_filename = os.path.basename(original_path)

            if focus_measure < threshold:
                destination_dir = original_blurry_dir
            else:
                destination_dir = original_clear_dir

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

            shutil.copy(original_path, os.path.join(destination_dir, original_filename))

        except Exception as e:
            print(f"Error processing {cropped_path}: {e}")

    num_blurry = len(os.listdir(original_blurry_dir)) - num_original_bad
    num_clear = len(os.listdir(original_clear_dir))

    # Save the annotations for clear images to a JSON file
    save_annotations(original_clear_annotations, original_clear_dir, "original_clear_annotations.json")

    return num_blurry, num_clear


# load the image mapping file, annotations, and process images as clear or blurry
def main(args):

    # LOad the image mapping 
    image_mapping = load_annotations(args.mapping_dir, "image_mapping.json")

    # Load the annotations
    annotations = load_annotations(args.input_dir, "coco_annotations_processed.json")

    num_blurry, num_clear = process_images(
        args.output_dir, image_mapping, args.threshold, annotations
    )
    print(f"Number of original blurry images: {num_blurry}")
    print(f"Number of original clear images: {num_clear}")


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
