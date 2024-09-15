import os
import shutil
import cv2
from processing_helperFunctions import load_annotations,save_annotations,delete_directory,create_output_directories

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


# Processes and classifies images as blurry or clear based on variance.
def process_images(output_dir, image_mapping, threshold, annotations,directories):
    uncropped_blurry_images_dir, uncropped_clear_images_dir = create_output_directories(output_dir,directories)
    original_clear_annotations = {"images": [], "annotations": []}

    for cropped_path, original_path in image_mapping.items():
        try:
            image = cv2.imread(cropped_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            focus_measure = variance_of_laplacian(gray)
            original_filename = os.path.basename(original_path)

            if focus_measure < threshold:
                uncropped_dir = uncropped_blurry_images_dir

            else:
                uncropped_dir = uncropped_clear_images_dir

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
            
            
            if not os.path.exists(os.path.join(uncropped_dir, original_filename)):
                shutil.copy(original_path, os.path.join(uncropped_dir, original_filename))


        except Exception as e:
            print(f"Error processing {cropped_path}: {e}")

    # Save the annotations for clear images to a JSON file
    save_annotations(original_clear_annotations, uncropped_clear_images_dir, "original_clear_annotations.json")

    print(f"Number of cropped blurry images:", len(os.listdir(uncropped_blurry_images_dir)))
    print(f"Number of cropped clear images:", len(os.listdir(uncropped_clear_images_dir)))



# load the image mapping file, annotations, and process images as clear or blurry
def main():

    # Load the image mapping 
    image_mapping = load_annotations(mapping_dir, "image_mapping.json")

    # Load the annotations
    annotations = load_annotations(input_dir, "coco_annotations_processed.json")

    process_images(
        output_dir, image_mapping, threshold, annotations,directories
    )

    # original cropped image directory is not needed anymore 
    delete_directory(mapping_dir)


 
if __name__ == "__main__":

    mapping_dir = "dataset/filtered_imgs/invalid_imgs/cropped_imgs"
    input_dir = "dataset/failed_crop_subset/"
    output_dir = "dataset/filtered_imgs/invalid_imgs"
    threshold = 60.0
    directories = ["uncropped_blurry_imgs","uncropped_clear_imgs"]
    
    main()
