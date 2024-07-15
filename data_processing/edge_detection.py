import os
import json
import shutil
import cv2
import argparse
from processing_helperFunctions import load_annotations, save_annotations


# Detects edges in an image using Canny edge detection.
def detect_edges(image_path, threshold1=100, threshold2=150):
    image = cv2.imread(image_path)  # read the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # convert to grayscale
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)  # reduce noise
    edges = cv2.Canny(blurred, threshold1, threshold2)  # apply canny

    return edges


# Checks if a bounding box is too close to the edge of the image.
def too_close_to_edge(bbox, image_shape, margin=1):

    # unpack bounding box coordinates
    left, top, width, height = bbox
    right, bottom = left + width, top + height

    # extract the image's height and width
    img_height, img_width = image_shape[:2]

    # returns true if the bounding box is too close to any image edge
    return (
        left <= margin
        or top <= margin
        or right >= img_width - margin
        or bottom >= img_height - margin
    )


# Chceks image size
def is_small_object(bbox, max_width=150, max_height=150):

    # Extracts bounded object's width and height
    width, height = bbox[2:]

    # Returns true if bounded image is less than the max width and height
    return width < max_width and height < max_height


# Create directories for storing fragmented images.
def create_output_directories(output_dir):
    bad_images_dir = os.path.join(output_dir, "bad_imgs")
    os.makedirs(bad_images_dir, exist_ok=True)
    return bad_images_dir


# Process images to isolate those with bounding boxes too close to the edges.
def process_images(input_dir, output_dir, annotations, margin):
    fragmented_dir = create_output_directories(os.path.dirname(output_dir))
    fragmented_annotations = {"images": [], "annotations": []}

    original_clear_annotations = annotations

    # Accessing the length of original_clear_images directory
    num_original_clear_images = len(os.listdir((input_dir)))
    print(f"Number of original clear images: {num_original_clear_images}")

    # Loop through each image's annotations
    for img in annotations["images"]:

        # Configure the image's path
        image_path = os.path.join(input_dir, img["file_name"])

        # detect edges for each image
        edges = detect_edges(image_path)

        # Extract the image's height and width
        image_shape = edges.shape

        # Creates a list that contains all th annotations for a specific image
        img_annotations = [
            ann for ann in annotations["annotations"] if ann["image_id"] == img["id"]
        ]

        # Loop through each annotation value
        for ann in img_annotations:

            # check if image is too close to the edge
            if too_close_to_edge(ann["bbox"], image_shape, margin) and is_small_object(
                ann["bbox"]
            ):
                fragmented_annotations["images"].append(
                    {"id": img["id"], "file_name": img["file_name"]}
                )
                fragmented_annotations["annotations"].append(ann)

                # moves fragmented images out of the clear directory
                shutil.move(image_path, os.path.join(fragmented_dir, img["file_name"]))

                # Remove the annotation from original clear annotations
                original_clear_annotations["images"] = [
                    image
                    for image in original_clear_annotations["images"]
                    if image["id"] != img["id"]
                ]
                original_clear_annotations["annotations"] = [
                    annotation
                    for annotation in original_clear_annotations["annotations"]
                    if annotation != ann
                ]

                break

    # Save the modified original clear and fragmented annotations
    save_annotations(fragmented_annotations, fragmented_dir, "fragmented_annotations.json")
    save_annotations(original_clear_annotations, input_dir,  "original_clear_annotations.json")

    num_new_clear_images = len(os.listdir(input_dir))

    # returns the number of fragmented images and the json file
    return num_original_clear_images, num_new_clear_images


def main(args):

    annotations = load_annotations(args.input_dir, "original_clear_annotations.json")
    num_original_clear_images, num_new_clear_images = process_images(
        args.input_dir, args.output_dir, annotations, args.margin
    )
    print(
        f"Total number of clear images after fragmented images removed: {num_new_clear_images}"
    )
    print(
        f"Number of fragmented images: { num_original_clear_images - num_new_clear_images}"
    )
    print(
        f"Total number of bad images after fragmented images added: {len(os.listdir(args.output_dir))}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input_dir",
        required=True,
        help="Path to input directory of original images and COCO file",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        required=True,
        help="Path to output directory of original images and COCO file",
    )

    parser.add_argument(
        "--margin",
        type=int,
        default=10,
        help="Margin threshold to consider bounding boxes too close to the edge",
    )
    args = parser.parse_args()

    main(args)
