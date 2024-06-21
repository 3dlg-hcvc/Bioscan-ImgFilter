import json
import shutil
import os
import argparse

from sklearn.model_selection import train_test_split


def create_directories(args):
    args.train_folder_path = os.path.join(args.dataset_name, "train")
    args.val_folder_path = os.path.join(args.dataset_name, "val")
    args.goodTrain_folder_path = os.path.join(args.train_folder_path, "good")
    args.badTrain_folder_path = os.path.join(args.train_folder_path, "bad")
    args.goodVal_folder_path = os.path.join(args.val_folder_path, "good")
    args.badVal_folder_path = os.path.join(args.val_folder_path, "bad")

    # Remove existing directories
    if os.path.exists(args.dataset_name):
        shutil.rmtree(args.dataset_name)

    os.makedirs(args.dataset_name, exist_ok=True)
    os.makedirs(args.train_folder_path, exist_ok=True)
    os.makedirs(args.val_folder_path, exist_ok=True)
    os.makedirs(args.goodTrain_folder_path, exist_ok=True)
    os.makedirs(args.badTrain_folder_path, exist_ok=True)
    os.makedirs(args.goodVal_folder_path, exist_ok=True)
    os.makedirs(args.badVal_folder_path, exist_ok=True)


def add_missing_information_to_coco_json(coco_annotation_dict):
    images = coco_annotation_dict["images"]
    annotations = coco_annotation_dict["annotations"]

    # Adding missing information to the annotation data
    for i in images:
        i["file_name"] = i["toras_path"][15:]
    for i in annotations:
        i["area"] = i["bbox"][2] * i["bbox"][3]
        i["iscrowd"] = 0
    return images, annotations


def remove_empty_annotations(coco_annotation_dict):
    images, annotations, categories = (
        coco_annotation_dict["images"],
        coco_annotation_dict["annotations"],
        coco_annotation_dict["categories"],
    )

    annotations = annotations[0 : len(images)]

    for idx, ann in enumerate(annotations):
        if ann["area"] == 0:
            images.pop(idx)
            annotations.pop(idx)

    for idx, ann in enumerate(annotations):
        ann["id"] = idx
        ann["image_id"] = idx
        images[idx]["id"] = idx

    return images, annotations, categories


def split_data_and_copy_image(args):
    coco_annotation_file = open(
        os.path.join(args.input_dir, "coco_annotations_processed.json")
    )
    coco_annotation_dict = json.load(coco_annotation_file)
    images, annotations, categories = remove_empty_annotations(coco_annotation_dict)

    # Separate images into "good" and "bad" based on annotations
    good_images, bad_images = [], []
    good_annotations, bad_annotations = [], []

    for img in images:
        curr_id = img["id"]
        if any(ann["image_id"] == curr_id and ann["area"] > 0 for ann in annotations):
            good_images.append(img)
        else:
            bad_images.append(img)

    for ann in annotations:
        if any(ann["image_id"] == img["id"] for img in good_images):
            good_annotations.append(ann)
        else:
            bad_annotations.append(ann)

    # Split "good" and "bad" images into training and validation sets
    good_train_ids, good_val_ids = train_test_split(
        [img["id"] for img in good_images], test_size=0.2
    )
    bad_train_ids, bad_val_ids = train_test_split(
        [img["id"] for img in bad_images], test_size=0.2
    )

    train_images, val_images = [], []
    train_annotations, val_annotations = [], []

    # Copy and split "good" images
    for img in good_images:
        curr_id = img["id"]
        src = os.path.join(args.input_dir, img["file_name"])
        if curr_id in good_train_ids:
            train_images.append(img)
            dst = os.path.join(args.goodTrain_folder_path, img["file_name"])
        else:
            val_images.append(img)
            dst = os.path.join(args.goodVal_folder_path, img["file_name"])
        shutil.copyfile(src, dst)

    # Copy and split "bad" images
    for img in bad_images:
        curr_id = img["id"]
        src = os.path.join(args.input_dir, img["file_name"])
        if curr_id in bad_train_ids:
            train_images.append(img)
            dst = os.path.join(args.badTrain_folder_path, img["file_name"])
        else:
            val_images.append(img)
            dst = os.path.join(args.badVal_folder_path, img["file_name"])
        shutil.copyfile(src, dst)

    # Split annotations into training and validation sets
    for ann in annotations:
        if ann["image_id"] in good_train_ids or ann["image_id"] in bad_train_ids:
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

    with open(os.path.join(args.train_folder_path, "custom_train.json"), "w") as f:
        json.dump(train_dict, f, indent=4)
    with open(os.path.join(args.val_folder_path, "custom_val.json"), "w") as f:
        json.dump(val_dict, f, indent=4)

    # Print the number of training and validation samples
    print(f"Number of training samples: {len(train_images)}")
    print(f"Number of validation samples: {len(val_images)}")


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

    create_directories(args)
    split_data_and_copy_image(args)
