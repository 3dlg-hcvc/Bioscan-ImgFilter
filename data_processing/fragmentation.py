import os
import shutil
import cv2
from processing_helperFunctions import load_annotations, save_annotations, move_directory, create_output_directories
from initial_filtering import compare_directories

# Detects edges in an image using Canny edge detection.
def detect_edges(image_path, threshold1=100, threshold2=150):
    image = cv2.imread(image_path)  # read the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # convert to grayscale
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)  # reduce noise
    edges = cv2.Canny(blurred, threshold1, threshold2)  # apply canny
    return edges

# Checks if a bounding box is too close to the edge of the image.
def too_close_to_edge(bbox, image_shape, margin=1):
    left, top, width, height = bbox
    right, bottom = left + width, top + height
    img_height, img_width = image_shape[:2]
    return (
        left <= margin
        or top <= margin
        or right >= img_width - margin
        or bottom >= img_height - margin
    )

# Checks image size
def is_small_object(bbox, max_width=100, max_height=100):
    width, height = bbox[2:]
    return width < max_width and height < max_height

# Process images to isolate those with bounding boxes too close to the edges.
def process_images(uncropped_clear_dir, output_dir, annotations, margin, directories):
    fragmented_dirs = create_output_directories(output_dir, directories)
    fragmented_dir = fragmented_dirs[0] if fragmented_dirs else ""

    fragmented_annotations = {"images": [], "annotations": []}
    original_clear_annotations = annotations.copy()

    
    for img in annotations["images"]:
        image_path = os.path.join(uncropped_clear_dir, img["file_name"])
        edges = detect_edges(image_path)
        image_shape = edges.shape

        img_annotations = [ann for ann in annotations["annotations"] if ann["image_id"] == img["id"]]

        for ann in img_annotations:
            if too_close_to_edge(ann["bbox"], image_shape, margin) and is_small_object(ann["bbox"]):
                fragmented_annotations["images"].append({"id": img["id"], "file_name": img["file_name"]})
                fragmented_annotations["annotations"].append(ann)

                fragmented_dest = os.path.join(fragmented_dir, img["file_name"])
                if not os.path.exists(fragmented_dest):
                    shutil.move(image_path, fragmented_dest)

                original_clear_annotations["images"] = [image for image in original_clear_annotations["images"] if image["id"] != img["id"]]
                original_clear_annotations["annotations"] = [annotation for annotation in original_clear_annotations["annotations"] if annotation != ann]

                break

    save_annotations(fragmented_annotations, fragmented_dir, "fragmented_annotations.json")
    
    # print(f"Total number of fragmented images:", len(os.listdir(fragmented_dir)))
    # print(f"Total number of valid images:", len(os.listdir(uncropped_clear_dir)))
    return len(os.listdir(fragmented_dir)),

def main():
    
    directories_paths = create_output_directories(output_dir, directories)
    fragmented_dir = directories_paths[0] if directories_paths else ""

    if len(os.listdir(fragmented_dir)) == 0:

        annotations = load_annotations(input_dir, "original_clear_annotations.json")
        margin = 50

        process_images(input_dir, fragmented_dir, annotations, margin, directories)
        move_directory(input_dir, dest_dir)

    else:
        print()

if __name__ == "__main__":
    input_dir = "dataset/filtered_imgs/invalid_imgs/uncropped_clear_imgs"
    output_dir = "dataset/filtered_imgs/invalid_imgs"
    dest_dir = "dataset/filtered_imgs/valid_imgs"
    directories = ["fragmented_imgs"]  # Make sure this is a list

    main()

