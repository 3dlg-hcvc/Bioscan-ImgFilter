
import shutil
import cv2
import os
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

from processing_helperFunctions import load_annotations, save_annotations
import os
import cv2
import numpy as np
import json
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from skimage.filters import sobel, roberts, laplace
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix



# Function to compute Laplacian variance
def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()


# Creates directories for storing processed original images.
def create_output_directories(output_dir):
    # Create good and bad image directories
    bad_images_dir = os.path.join(output_dir, "bad_imgs")
    good_images_dir = os.path.join(output_dir, "good_imgs")
    #blurry_images_dir = os.path.join(output_dir, "blurry_imgs")
    #clear_images_dir = os.path.join(output_dir, "clear_imgs")
    cropped_blurry_images_dir = os.path.join(output_dir, "cropped_blurry_imgs")
    cropped_clear_images_dir = os.path.join(output_dir, "cropped_clear_imgs")

    #os.makedirs(blurry_images_dir, exist_ok=True)
    #os.makedirs(clear_images_dir, exist_ok=True)
    os.makedirs(cropped_blurry_images_dir, exist_ok=True)
    os.makedirs(cropped_clear_images_dir, exist_ok=True)

    return bad_images_dir, good_images_dir, cropped_blurry_images_dir, cropped_clear_images_dir

# Function to load images and their labels
def load_images_and_labels(clear_dir, blurry_dir):
    images = []
    labels = []
    
    # Load clear images
    for filename in os.listdir(clear_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(clear_dir, filename)
            images.append(img_path)
            labels.append(1)  # Label 1 for clear images

    # Load blurry images
    for filename in os.listdir(blurry_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(blurry_dir, filename)
            images.append(img_path)
            labels.append(0)  # Label 0 for blurry images

    return images, labels

# Function to extract features using edge detection
def extract_features(image_paths):
    features = []
    for img_path in image_paths:
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is not None:
            sobel_edges = sobel(image)
            roberts_edges = roberts(image)
            laplacian_edges = laplace(image)
            
            features.append([
                sobel_edges.var(), sobel_edges.mean(), sobel_edges.max(),
                roberts_edges.var(), roberts_edges.mean(), roberts_edges.max(),
                laplacian_edges.var(), laplacian_edges.mean(), laplacian_edges.max()
            ])
    return np.array(features)

# Function to load image mappings from a JSON file
def load_image_mappings(mapping_file):
    with open(mapping_file, 'r') as f:
        return json.load(f)
    

# Function to save annotations to a JSON file
def save_annotations(annotations, directory, filename):
    with open(os.path.join(directory, filename), 'w') as f:
        json.dump(annotations, f)

# Function to classify all cropped images from "crop_imgs" and save images into good/bad and cropped clear/blurry directories 
def classify_and_save_images(model, output_dir, image_mapping, annotations):
    bad_dir, good_dir, blurry_dir, clear_dir = create_output_directories(output_dir)
    original_clear_annotations = {"images": [], "annotations": []}

    for cropped_path, original_path in image_mapping.items():
        #cropped_path_full = os.path.join(output_dir, "cropped_imgs")
        features = extract_features([cropped_path])
        original_filename = os.path.basename(original_path)
        
        if features.size == 0:
            continue
        
        prediction = model.predict(features)[0]
        
        if prediction == 1:
            dest_dir = good_dir
            subset_dir = clear_dir


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
        else:
            dest_dir = bad_dir
            subset_dir = blurry_dir
        
        # Save original image in the respective good/bad directory
        if not os.path.exists(os.path.join(dest_dir, os.path.basename(original_path))):
            shutil.copy(original_path, os.path.join(dest_dir, os.path.basename(original_path)))
        
        # Save cropped image in the respective cropped clear/blurry directory
        if not os.path.exists(os.path.join(subset_dir, os.path.basename(cropped_path))):
            shutil.copy(cropped_path, os.path.join(subset_dir, os.path.basename(cropped_path)))

    save_annotations(original_clear_annotations, good_dir, "original_clear_annotations.json")
    save_annotations(original_clear_annotations, clear_dir, "original_clear_annotations.json")
    print(f"\nNumber of cropped blurry images:", len(os.listdir(blurry_dir)))
    print(f"Number of cropped clear images:", len(os.listdir(clear_dir)))
    #print(f"Number of bad images:", len(os.listdir(bad_dir)))
    #print(f"Number of good images:", len(os.listdir(good_dir)))


# Function to evaluate the SVM model with edge detection features
def evaluate_svm_model(clear_dir, blurry_dir):
    image_paths, labels = load_images_and_labels(clear_dir, blurry_dir)
    
    # Split data into train and test sets
    train_paths, test_paths, train_labels, test_labels = train_test_split(image_paths, labels, test_size=0.4, random_state=42)
    print(len(image_paths))

    print(len(train_paths))
    print(len(test_paths))
    
    # Extract features
    train_features = extract_features(train_paths)
    test_features = extract_features(test_paths)
    
    # Define the parameter grid for SVM
    param_grid = {
        'C': [0.1, 1, 10],
        'gamma': [1, 0.1, 0.01],
        'kernel': ['rbf', 'poly', 'sigmoid']
    }
    
    # Create an SVM classifier
    svm = SVC()
    
    # Perform grid search
    grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(train_features, train_labels)
    
    # Get the best model
    best_svm = grid_search.best_estimator_
    
    # Predict on the test set
    pred_labels = best_svm.predict(test_features)
    
    # Compute performance metrics
    accuracy = accuracy_score(test_labels, pred_labels)
    precision = precision_score(test_labels, pred_labels)
    recall = recall_score(test_labels, pred_labels)
    f1 = f1_score(test_labels, pred_labels)

    # Compute confusion matrix
    conf_matrix = confusion_matrix(test_labels, pred_labels)
    class_report = classification_report(test_labels, pred_labels, target_names=['Blurry', 'Clear'])

    # Print results
    print("SVM Classifier with Edge Detection Features:")
    print("Best Parameters:", grid_search.best_params_)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("\nConfusion Matrix:")
    print(conf_matrix)
    print("\nClassification Report:")
    print(class_report)


    # Save misclassified images
    misclassified_dir = 'misclassified_imgs_cropped'
    os.makedirs(misclassified_dir, exist_ok=True)

    for img_path, true_label, pred_label in zip(test_paths, test_labels, pred_labels):
       if true_label != pred_label:
          img = cv2.imread(img_path)
          if img is not None:
            misclassified_filename = f"{os.path.basename(img_path)}_true_{true_label}_pred_{pred_label}.jpg"
            misclassified_path = os.path.join(misclassified_dir, misclassified_filename)
            cv2.imwrite(misclassified_path, img)

    joblib.dump(best_svm, 'best_svm_model.pkl')

    

# Creates directories for storing processed original images.
def create_output_directories_uncropped_imgs(subdataset_dir):

    # Create good and bad image directories
    uncropped_blurry_images_dir = os.path.join(subdataset_dir, "uncropped_blurry_imgs_labelled")
    uncropped_clear_images_dir = os.path.join(subdataset_dir, "uncropped_clear_imgs_labelled")

    os.makedirs(uncropped_blurry_images_dir, exist_ok=True)
    os.makedirs(uncropped_clear_images_dir, exist_ok=True)

    return uncropped_blurry_images_dir, uncropped_clear_images_dir


def store_labelled_uncropped_imgs(clear_images_dir, blurry_images_dir, image_mapping, subdataset_dir):
    # Make uncropped directories 
    uncropped_blurry_dir, uncropped_clear_dir = create_output_directories_uncropped_imgs(subdataset_dir)

    # Get the list of files in the directories
    clear_files = os.listdir(clear_images_dir)
    blurry_files = os.listdir(blurry_images_dir)

    for cropped_path, original_path in image_mapping.items():
        original_filename = os.path.basename(original_path)

        if original_filename in clear_files and not os.path.exists(os.path.join(uncropped_clear_dir, original_filename)):
            shutil.copy(original_path, os.path.join(uncropped_clear_dir, original_filename))
 
        elif original_filename in blurry_files and not os.path.exists(os.path.join(uncropped_blurry_dir, original_filename)):
            shutil.copy(original_path, os.path.join(uncropped_blurry_dir, original_filename))
        else:
            continue 

    print(f"\nNumber of uncropped labelled blurry images:", len(os.listdir(uncropped_blurry_dir)))
    print(f"Number of uncropped labelled clear images:", len(os.listdir(uncropped_clear_dir)))

        


if __name__ == "__main__":
    # Define the directories containing images
    clear_images_dir = 'dataset/cropped_clear_imgs_manual'
    blurry_images_dir = 'dataset/cropped_blurry_imgs_manual'

    print("NUMBER OF CLEAR LABELLED IMAGES: ",len(os.listdir(clear_images_dir)))
    print("NUMBER OF BLURRY LABELLED IMAGES",len(os.listdir(blurry_images_dir)))


    # Evaluate the SVM model with edge detection features
    evaluate_svm_model(clear_images_dir, blurry_images_dir)
    
    # Load the saved best model
    best_model = joblib.load('best_svm_model.pkl')

    # Input directory for cropped images
    cropped_images_dir = 'dataset/cropped_imgs'

    # Output directory for classified images
    output_dir = "dataset/processed_imgs"

    subdataset_dir = "dataset/"

    # Load image mappings from JSON file
    mapping_file = 'dataset/processed_imgs/cropped_imgs'
    input_file = "dataset/failed_crop_subset"
    image_mapping = load_annotations(mapping_file, "image_mapping.json")
    annotations = load_annotations(input_file, "coco_annotations_processed.json")

    store_labelled_uncropped_imgs(clear_images_dir, blurry_images_dir, image_mapping, subdataset_dir)

    # Classify and save images
    classify_and_save_images(best_model, output_dir, image_mapping,annotations)


