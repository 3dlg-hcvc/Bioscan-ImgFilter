import json
import shutil
import os
import argparse
from sklearn.model_selection import train_test_split

def create_output_directories(output_dir, dataset_name):
    train_folder_path = os.path.join(dataset_name, "train")
    good_train_folder_path = os.path.join(train_folder_path, "good")
    bad_train_folder_path = os.path.join(train_folder_path, "bad")

    val_folder_path = os.path.join(dataset_name, "val")
    good_val_folder_path = os.path.join(val_folder_path, "good")
    bad_val_folder_path = os.path.join(val_folder_path, "bad")

    os.makedirs(good_train_folder_path, exist_ok=True)
    os.makedirs(bad_train_folder_path, exist_ok=True)
    os.makedirs(good_val_folder_path, exist_ok=True)
    os.makedirs(bad_val_folder_path, exist_ok=True)

    return good_train_folder_path, bad_train_folder_path, good_val_folder_path, bad_val_folder_path

def add_missing_information_to_coco_json(coco_annotation_dict):
    images = coco_annotation_dict["images"]
    annotations = coco_annotation_dict["annotations"]

    # Adding missing information to the annotation data
    for i in images:
        if "file_name" not in i:
            i["file_name"] = i["coco_url"].split('/')[-1]
    for i in annotations:
        i["area"] = i["bbox"][2] * i["bbox"][3]
        i["iscrowd"] = 0
    return images, annotations

def idx_to_annotations(coco_annotation_dict):
    images, annotations, categories = (
        coco_annotation_dict["images"],
        coco_annotation_dict["annotations"],
        coco_annotation_dict["categories"],
    )

    for idx, ann in enumerate(annotations):
        ann["id"] = idx
        ann["image_id"] = idx
        images[idx]["id"] = idx

    return images, annotations, categories

def split_data(args):
    # Load Coco annotations  
    with open(os.path.join(args.input_dir, "coco_annotations_processed.json")) as coco_annotation_file:
        coco_annotation_dict = json.load(coco_annotation_file)
    
    images, annotations, categories = idx_to_annotations(coco_annotation_dict)

    good_train_folder_path, bad_train_folder_path, good_val_folder_path, bad_val_folder_path = create_output_directories(args.input_dir, args.dataset_name)

    # Gather all image file names from good and bad directories
    good_images = [f for f in os.listdir(os.path.join(args.dataset_name, "good_imgs")) if f.endswith('.jpg')]
    bad_images = [f for f in os.listdir(os.path.join(args.dataset_name, "bad_imgs")) if f.endswith('.jpg')]

    # Split "good" and "bad" images into training and validation sets
    good_train_imgs, good_val_imgs = train_test_split(good_images, test_size=0.2)
    bad_train_imgs, bad_val_imgs = train_test_split(bad_images, test_size=0.2)

    train_images, val_images = [], []
    train_annotations, val_annotations = [], []

    # Copy and split "good" images
    for img in good_images:
        src = os.path.join(args.dataset_name, "good_imgs", img)
        if img in good_train_imgs:
            dst = os.path.join(good_train_folder_path, img)
            train_images.append({"file_name": img, "id": len(train_images)})
        else:
            dst = os.path.join(good_val_folder_path, img)
            val_images.append({"file_name": img, "id": len(val_images) + len(train_images)})
        shutil.copyfile(src, dst)

    # Copy and split "bad" images
    for img in bad_images:
        src = os.path.join(args.dataset_name, "bad_imgs", img)
        if img in bad_train_imgs:
            dst = os.path.join(bad_train_folder_path, img)
            train_images.append({"file_name": img, "id": len(train_images)})
        else:
            dst = os.path.join(bad_val_folder_path, img)
            val_images.append({"file_name": img, "id": len(val_images) + len(train_images)})
        shutil.copyfile(src, dst)

    # Split annotations into training and validation sets
    for ann in annotations:
        if ann["image_id"] in [img["id"] for img in train_images]:
            train_annotations.append(ann)
        else:
            val_annotations.append(ann)

    # Save annotation data
    train_dict = {
        "images": train_images,
        "categories": categories,
        "annotations": train_annotations,
    }
    val_dict = {
        "images": val_images,
        "categories": categories,
        "annotations": val_annotations,
    }

    with open(os.path.join(good_train_folder_path, "custom_train.json"), "w") as f:
        json.dump(train_dict, f, indent=4)
    with open(os.path.join(good_val_folder_path, "custom_val.json"), "w") as f:
        json.dump(val_dict, f, indent=4)

    # Print the number of training and validation samples
    print(f"Number of training samples: {len(train_images)}")
    print(f"Number of validation samples: {len(val_images)}")

    # Calculate and print the number of "good" and "bad" images in the training set
    num_good_train = len(os.listdir(good_train_folder_path))
    num_bad_train = len(os.listdir(bad_train_folder_path))
    num_good_val = len(os.listdir(good_val_folder_path))
    num_bad_val = len(os.listdir(bad_val_folder_path))

    print(f"Number of good images in train: {num_good_train}")
    print(f"Number of bad images in train: {num_bad_train}")
    print(f"Number of good images in validation: {num_good_val}")
    print(f"Number of bad images in validation: {num_bad_val}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to the folder that contains the images and coco file.",
    )
    parser.add_argument(
        "--dataset_name", type=str, required=True, help="Name of the dataset"
    )
 
 
    args = parser.parse_args()

    split_data(args)
