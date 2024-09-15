import os
import shutil
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image

from processing_helperFunctions import load_annotations, augment_image

def create_output_directories(output_dir):

    # create good and bad image directories
    # test_dir = os.path.join(output_dir, "test_dir")
    # os.makedirs(test_dir, exist_ok=True)

    # denoised_dir = os.path.join(output_dir, "denoised_dir")
    # os.makedirs(denoised_dir, exist_ok=True)

    orig_blur_dir = os.path.join(output_dir, "uncropped_blurry_imgs_labelled")
    os.makedirs(orig_blur_dir, exist_ok=True)


    orig_clear_dir = os.path.join(output_dir, "uncropped_clear_imgs_labelled")
    os.makedirs(orig_clear_dir, exist_ok=True)

    return orig_blur_dir, orig_clear_dir


def load_image(image_path):
    """Load the image from the given path."""
    return cv2.imread(image_path)


def is_unbounded(image):
    """Check if the image is unbounded by evaluating its edges."""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply a binary threshold to emphasize the edges
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Check if the image has significant black and white regions (e.g., circular pattern)
    edges = cv2.Canny(binary, 100, 200)
    edge_density = np.sum(edges) / edges.size

    # Assuming unbounded means the edge density is low (no clear boundary)
    if edge_density < 0.01:  # This threshold may need tuning based on images
        return True
    return False


def check_if_unbounded_dir(input_dir, destination_dir, mapping_dir, image_mapping):
    print("Original unbounded: ", len(os.listdir(destination_dir)))

    for cropped_img_name in os.listdir(input_dir):
        cropped_img_path = os.path.join(input_dir, cropped_img_name)
        cropped_filename = os.path.basename(cropped_img_path)
        image = load_image(cropped_img_path)
        
        if is_unbounded(image):
            # Find the corresponding original filename using the image mapping
            for cropped_path, original_path in image_mapping.items():
                if cropped_filename in original_path and not os.path.exists(os.path.join(destination_dir, os.path.basename(original_path))):
                    shutil.copy(original_path, os.path.join(destination_dir, os.path.basename(original_path)))

    print("Filtered unbounded: ", len(os.listdir(destination_dir)))



def crop_to_original(cropped_img_dir, uncropped_dir, mapping_dir, image_mapping):
    print("cropped dir length: ", len(os.listdir(uncropped_dir)))

    for cropped_img_name in os.listdir(cropped_img_dir):
        cropped_img_path = os.path.join(cropped_img_dir, cropped_img_name)
        cropped_filename = os.path.basename(cropped_img_path)
        image = load_image(cropped_img_path)
        
        for cropped_path, original_path in image_mapping.items():
            if cropped_filename in original_path and not os.path.exists(os.path.join(uncropped_dir, os.path.basename(original_path))):
                shutil.copy(original_path, os.path.join(uncropped_dir, os.path.basename(original_path)))

    print("uncropped dir length: ", len(os.listdir(uncropped_dir)))


def compare_directories(input_dir, output_dir):
    print("original dir length: ",len(os.listdir(output_dir)))
    for img in os.listdir(input_dir):
        img_path = os.path.join(input_dir, img)
        filename = os.path.basename(img_path)
        
        # Check if the file exists in the output  directory
        if filename in os.listdir(output_dir):
            output_img_path = os.path.join(output_dir, filename)
            
            # Delete the matching file from the unboundeoutput directory
            if os.path.isfile(output_img_path):
                os.remove(output_img_path)

    print("filtered output dir length: ",len(os.listdir(output_dir)))
    print("filtered input dir length: ",len(os.listdir(input_dir)))


import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


# def remove_specks(image, min_size=5000):
#     """Remove small specks from the binary image."""
#     # Label connected components
#     num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(image, connectivity=8)

#     # Remove small regions
#     for i in range(1, num_labels):
#         if stats[i, cv2.CC_STAT_AREA] < min_size:
#             image[labels == i] = 0

#     return image


# def denoise(unbounded_dir, denoised_dir):
#     for img_name in os.listdir(unbounded_dir):
#         unbounded_img_path = os.path.join(unbounded_dir, img_name)

#         # Load the image
#         img = load_image(unbounded_img_path)

#         # Convert to grayscale
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#         # Apply a binary threshold
#         _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

#         # Remove small specks using morphological operations
#         cleaned = remove_specks(binary)

#         # Apply morphological opening to remove noise
#         kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
#         cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

#         # Invert back to original state if needed
#         cleaned = cv2.bitwise_not(cleaned)

#         # Apply fast non-local means denoising
#         noiseless_image_colored = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

#         # Save the cleaned image
#         denoised_img_path = os.path.join(denoised_dir, img_name)
#         cv2.imwrite(denoised_img_path, noiseless_image_colored)





# def augment_img_dir(frag_dir, output_dir, num_augmentations_per_image=2):
#     """Apply augmentations to images in the fragmented images directory and save them."""
#     os.makedirs(output_dir, exist_ok=True)

#     print(" dir original length: ", len(os.listdir(frag_dir)))
    
#     for img_name in os.listdir(frag_dir):
#         img_path = os.path.join(frag_dir, img_name)
#         augment_image(img_path, output_dir, num_augmentations=num_augmentations_per_image)
    
#     print("Augmentation complete. Augmented images saved to:", output_dir)
    
# def fix_manual(frag_manual_dir, failed_crop_dir):
#     for img_name in os.listdir(frag_manual_dir):
#         # Check if the image is an augmented image (contains "_aug_")
#         if "_aug_" in img_name:
#             # Get the base image name by removing the augmentation suffix
#             base_name = img_name.split('_aug_')[0] + ".jpg"
            
#             # Construct the path to the base image in the failed crop directory
#             failed_crop_path = os.path.join(failed_crop_dir, base_name)
            
#             # If the base image exists in the failed crop directory
#             if os.path.exists(failed_crop_path):
#                 # Remove the augmented image from frag_manual_dir
#                 img_path = os.path.join(frag_manual_dir, img_name)
#                 os.remove(img_path)
#                 print(f"Removed {img_name} from {frag_manual_dir}")
                
#                 # Copy the failed crop image to the frag_manual_dir
#                 shutil.copy(failed_crop_path, frag_manual_dir)
#                 print(f"Copied {base_name} from {failed_crop_dir} to {frag_manual_dir}")

def main():
    # Load your images
    # cropped_blur_dir = "dataset/processed_imgs/cropped_blurry_imgs"
    # cropped_clear_dir = "dataset/processed_imgs/cropped_clear_imgs"
    # test_dir = "dataset/processed_imgs/test_dir"
    # mapping_dir = "dataset/processed_imgs/cropped_imgs"
    # output_dir = "dataset/processed_imgs"
    frag_dir = "dataset/filtered_imgs/invalid_imgs/fragmented_imgs"
    blur_unc = "dataset/filtered_imgs/invalid_imgs/uncropped_blurry_imgs"
    empty_imgs = "dataset/filtered_imgs/invalid_imgs/empty_imgs"
    #unbounded_dir = "dataset/processed_imgs/unbounded_imgs"
    valid = "dataset/filtered_imgs/valid"

    #denoised_dir = create_output_directories(output_dir)
    #orig_blur_dir, orig_clear_dir = create_output_directories(output_dir)


    #image_mapping = load_annotations(mapping_dir, "image_mapping.json")
    #check_if_unbounded_dir(input_dir, test_dir,mapping_dir,image_mapping)

    #compare_directories(cropped_blur_dir, cropped_clear_dir)
    #denoise(unbounded_dir, denoised_dir)
    #augment_fragments(frag_dir, frag_dir, num_augmentations_per_image=2)


    #crop_to_original(cropped_blur_dir, orig_blur_dir, mapping_dir, image_mapping)

    #crop_to_original(cropped_clear_dir, orig_clear_dir, mapping_dir, image_mapping)
    compare_directories(valid,blur_unc )


    #frag_manual_dir = "dataset/processed_imgs/frag_manual"
    #failed_crop_dir = "dataset/failed_crop_subset"
    # fix_manual(frag_manual_dir, failed_crop_dir)



        
   

if __name__ == "__main__":
    main()
