import os
import json
import shutil
import cv2
import argparse

# Detects edges in an image using Canny edge detection.
def detect_edges(image_path, threshold1=100, threshold2=150):
    image = cv2.imread(image_path) # read the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # convert to grayscale
    blurred = cv2.GaussianBlur(gray, (3, 3), 0) # reduce noise 
    edges = cv2.Canny(blurred, threshold1, threshold2)  # apply canny 

    #cv2.imshow('Canny Edge Detection', edges)
    #cv2.waitKey(0)
    return edges

# Checks if a bounding box is too close to the edge of the image.
def too_close_to_edge(bbox, image_shape, margin=1):

    # unpack bounding box coordinates 
    left, top, width, height = bbox 
    right, bottom = left + width, top + height

    # extract the image's height and width  
    img_height, img_width = image_shape[:2] 

    # Check if the bounding box is too close to any edge of the image
    # Return True if the bounding box is within 'margin' pixels of the left, top, right, or bottom edges of the image
    return (left <= margin or top <= margin or right >= img_width - margin or bottom >= img_height - margin) 


# Chceks image size 
def is_small_object(bbox, max_width=150, max_height=150):

    # Extracts bounded object's width and height 
    width,height = bbox[2:]
    
    # Returns true if bounded image is less than the max width and height parameters 
    return (width <max_width and height <max_height)


# Save updated annotations back to JSON file.
def save_annotations(annotations, output_file):
    with open(output_file, "w") as f:
        json.dump(annotations, f, indent=4)


# Load the "clear" annotations from the JSON file.
def load_annotations(input_dir):
    with open(os.path.join(input_dir, "original_clear_annotations.json")) as f: # loads clear annotations 
        return json.load(f)

# Create directories for storing fragmented images.
def create_output_directories(output_dir):
    bad_images_dir = os.path.join(output_dir, "bad_imgs")
    #fragmented_dir = os.path.join(bad_images_dir, "fragmented_images")
    #os.makedirs(fragmented_dir, exist_ok=True)
    return bad_images_dir

# Save updated annotations back to JSON file.
def save_annotations(annotations, output_file):
    with open(output_file, "w") as f:
        json.dump(annotations, f, indent=4)

# Process images to isolate those with bounding boxes too close to the edges.
def process_images(input_dir, output_dir, annotations, margin):
    fragmented_dir = create_output_directories(os.path.dirname(output_dir))
    too_close_annotations = {"images": [], "annotations": []}

    original_clear_annotations = annotations

    # Accessing the length of original_clear_images directory
    original_clear_images_dir = os.path.join(os.path.dirname(input_dir), "original_clear_images")
    num_original_clear_images = len(os.listdir(original_clear_images_dir))
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
        img_annotations = [ann for ann in annotations["annotations"] if ann["image_id"] == img["id"]]

        # Loop through each annotation value 
        for ann in img_annotations:

            # check if image is too close to the edge 
            if too_close_to_edge(ann["bbox"], image_shape, margin) and is_small_object(ann["bbox"]):
                too_close_annotations["images"].append({
                    "id": img["id"],
                    "file_name": img["file_name"]
                })
                too_close_annotations["annotations"].append(ann)

                # moves fragmented images out of the clear directory 
                #print("iss fragment")
                shutil.move(image_path, os.path.join(fragmented_dir, img["file_name"]))

                # Remove the annotation from original clear annotations
                original_clear_annotations["images"] = [
                    image for image in original_clear_annotations["images"] if image["id"] != img["id"]
                ]
                original_clear_annotations["annotations"] = [
                    annotation for annotation in original_clear_annotations["annotations"] if annotation != ann
                ]
            
                break
    # crates a json file containing annotations for images too close to the edge 
    too_close_annotations_file = os.path.join((fragmented_dir), "fragmented_annotations.json")
    with open(too_close_annotations_file, "w") as f:
        json.dump(too_close_annotations, f, indent=4)

    # Save the modified original clear annotations back to file in original_clear_images directory
    original_clear_annotations_file = os.path.join(original_clear_images_dir, "original_clear_annotations.json")
    save_annotations(original_clear_annotations, original_clear_annotations_file)


    num_original_clear_images = len(os.listdir(original_clear_images_dir))
    print(f"Number of original clear images after fragmented images removed: {num_original_clear_images}")

    # returns the number of fragmented images and the json file 
    return len(os.listdir(fragmented_dir)), too_close_annotations_file

def main(args):

    #with open(os.path.join(args.input_dir, "original_clear_annotations.json")) as f: # loads clear annotations 
    #    annotations = json.load(f)
    annotations = load_annotations(args.input_dir)
    num_fragmented, too_close_annotations_file = process_images(args.input_dir, args.output_dir, annotations, args.margin)
    print(f"Number of images with bounding boxes too close to the edge: {num_fragmented}")
    print(f"Images with bounding boxes too close to the edge and their annotations saved to: {too_close_annotations_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", required=True, help="Path to input directory of original images and COCO file")
    parser.add_argument("-o", "--output_dir", required=True, help="Path to output directory of original images and COCO file")

    parser.add_argument("--margin", type=int, default=10, help="Margin threshold to consider bounding boxes too close to the edge")
    args = parser.parse_args()

    main(args)
