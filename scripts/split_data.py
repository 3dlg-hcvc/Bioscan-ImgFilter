import json
import shutil
import os
import argparse
from sklearn.model_selection import train_test_split

def create_output_directories(output_dir):
    # Create training, good/bad directories
    train_folder_path = os.path.join(output_dir, "train")
    good_train_folder_path = os.path.join(train_folder_path, "good")
    bad_train_folder_path = os.path.join(train_folder_path, "bad")

    # Create validation, good/bad directories
    val_folder_path = os.path.join(output_dir, "val")
    good_val_folder_path = os.path.join(val_folder_path, "good")
    bad_val_folder_path = os.path.join(val_folder_path, "bad")

    # Create train, clear/blurry directories
    train_blur_detection = os.path.join(output_dir, "uncropped_train_blur_detection")
    unc_train_clear_path = os.path.join(train_blur_detection, "clear")
    unc_train_blurry_path = os.path.join(train_blur_detection, "blurry")

    # Create validation, clear/blurry directories
    val_blur_detection = os.path.join(output_dir, "uncropped_val_blur_detection")
    val_clear_path = os.path.join(val_blur_detection, "uncropped_clear")
    val_blurry_path = os.path.join(val_blur_detection, "uncropped_blurry")


    # Create train, lab clear/blurry directories
    lab_train_blur_detection = os.path.join(output_dir, "cropped_train_blur_detection")
    cropped_train_clear_path = os.path.join(lab_train_blur_detection, "crop_clear")
    cropped_train_blurry_path = os.path.join(lab_train_blur_detection, "crop_blurry")

    # Create validation, clear/blurry directories
    lab_val_blur_detection = os.path.join(output_dir, "cropped_val_blur_detection")
    lab_val_clear_path = os.path.join(lab_val_blur_detection, "crop_clear")
    lab_val_blurry_path = os.path.join(lab_val_blur_detection, "crop_blurry")

    # Remove and reset directories if they previously existed
    if os.path.exists(train_folder_path) or os.path.exists(val_folder_path):
        shutil.rmtree(train_folder_path)
        shutil.rmtree(val_folder_path)

    # Remove and reset directories if they previously existed
    if os.path.exists(val_blur_detection) or os.path.exists(train_blur_detection):
        shutil.rmtree(train_blur_detection)
        shutil.rmtree(val_blur_detection)

    # Remove and reset directories if they previously existed
    if os.path.exists(lab_val_blur_detection) or os.path.exists(lab_train_blur_detection):
        shutil.rmtree(lab_train_blur_detection)
        shutil.rmtree(lab_val_blur_detection)

    os.makedirs(good_train_folder_path, exist_ok=True)
    os.makedirs(bad_train_folder_path, exist_ok=True)
    os.makedirs(good_val_folder_path, exist_ok=True)
    os.makedirs(bad_val_folder_path, exist_ok=True)

    os.makedirs(unc_train_clear_path, exist_ok=True)
    os.makedirs(unc_train_blurry_path, exist_ok=True)
    os.makedirs(val_clear_path, exist_ok=True)
    os.makedirs(val_blurry_path, exist_ok=True)

    os.makedirs(cropped_train_clear_path, exist_ok=True)
    os.makedirs(cropped_train_blurry_path, exist_ok=True)
    os.makedirs(lab_val_clear_path, exist_ok=True)
    os.makedirs(lab_val_blurry_path, exist_ok=True)

    return (
        good_train_folder_path,
        bad_train_folder_path,
        good_val_folder_path,
        bad_val_folder_path,

        unc_train_clear_path,
        unc_train_blurry_path,
        val_clear_path,
        val_blurry_path,

        cropped_train_clear_path,
        cropped_train_blurry_path,
        lab_val_clear_path,
        lab_val_blurry_path
    )

# Organize and vcategorize annotations
def add_missing_information_to_coco_json(coco_annotation_dict):
    images = coco_annotation_dict["images"]
    annotations = coco_annotation_dict["annotations"]
    # Adding missing information to the annotation data
    for i in images:
        if "file_name" not in i:
            i["file_name"] = i["coco_url"].split("/")[-1]
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

# Load all image annotations
def load_annotations(input_dir):
    with open(os.path.join(input_dir, "coco_annotations_processed.json")) as f:
        return json.load(f)
    
def split_data(args):
    # Load Coco annotations
    coco_annotation_dict = load_annotations(args.input_dir)
    images, annotations, categories = idx_to_annotations(coco_annotation_dict)
    (
        good_train_folder_path,
        bad_train_folder_path,
        good_val_folder_path,
        bad_val_folder_path,

        unc_train_clear_path,
        unc_train_blurry_path,
        val_clear_path,
        val_blurry_path,

        cropped_train_clear_path,
        cropped_train_blurry_path,
        lab_val_clear_path,
        lab_val_blurry_path

    ) = create_output_directories(args.output_dir)
    # Gather all image file names from good and bad directories
    good_images = [
        f
        for f in os.listdir(os.path.join(args.dataset_name, "good_imgs"))
        if f.endswith(".jpg")
    ]
    bad_images = [
        f
        for f in os.listdir(os.path.join(args.dataset_name, "bad_imgs"))
        if f.endswith(".jpg")
    ]
    unc_blurry_images = [
        f
        for f in os.listdir(os.path.join(args.dataset_name, "uncropped_blurry_imgs_labelled"))
        if f.endswith(".jpg")
    ]
    unc_clear_images = [
        f
        for f in os.listdir(os.path.join(args.dataset_name, "uncropped_clear_imgs_labelled"))
        if f.endswith(".jpg")
    ]

    crop_clear_imgs = [
        f
        for f in os.listdir(os.path.join(args.dataset_name, "cropped_clear_imgs_manual"))
        if f.endswith(".jpg")
    ]

    crop_blurry_imgs = [
        f
        for f in os.listdir(os.path.join(args.dataset_name, "cropped_blurry_imgs_manual"))
        if f.endswith(".jpg")
    ]

    num_cropped_blurry_imgs = len((crop_blurry_imgs))
    num_cropped_clear_imgs = len((crop_clear_imgs))
    num_uncropped_blurry_imgs = len((unc_blurry_images))
    num_uncropped_clear_imgs = len((unc_clear_images))

    print("Total number of CROPPED blurry images",num_cropped_blurry_imgs)
    print("TOtal number of CROPPED clear images: ",num_cropped_clear_imgs)

    print("\nTotal number of UNCROPPED blurry images: ",num_uncropped_blurry_imgs)
    print("TOtal number of CROPPED clear images: ",num_uncropped_clear_imgs)



    # Split "good" and "bad" images into training and validation sets
    good_train_imgs, good_val_imgs = train_test_split(good_images, test_size=0.2)
    bad_train_imgs, bad_val_imgs = train_test_split(bad_images, test_size=0.2)

    # Split uncropped clear/blurry images into training and validation sets
    clear_train_imgs, good_val_imgs = train_test_split(unc_clear_images, test_size=0.4)
    blurry_train_imgs, bad_val_imgs = train_test_split(unc_blurry_images, test_size=0.4)

    # Split cropped clear/blurry images into training and validation sets
    lab_clear_train_imgs, lab_clear_val_imgs = train_test_split(crop_clear_imgs, test_size=0.4)
    lab_blurry_train_imgs, lab_blurry_val_imgs = train_test_split(crop_blurry_imgs, test_size=0.4)

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
            val_images.append(
                {"file_name": img, "id": len(val_images) + len(train_images)}
            )
        shutil.copyfile(src, dst)

    # Copy and split "bad" images
    for img in bad_images:
        src = os.path.join(args.dataset_name, "bad_imgs", img)
        if img in bad_train_imgs:
            dst = os.path.join(bad_train_folder_path, img)
            train_images.append({"file_name": img, "id": len(train_images)})
        else:
            dst = os.path.join(bad_val_folder_path, img)
            val_images.append(
                {"file_name": img, "id": len(val_images) + len(train_images)}
            )
        shutil.copyfile(src, dst)


    # Copy and split uncropped "blurry" images
    for img in unc_blurry_images:
        src = os.path.join(args.dataset_name, "uncropped_blurry_imgs_labelled", img)
        if img in blurry_train_imgs:
            dst = os.path.join(unc_train_blurry_path, img)
            train_images.append({"file_name": img, "id": len(train_images)})
        else:
            dst = os.path.join(val_blurry_path, img)
            val_images.append(
                {"file_name": img, "id": len(val_images) + len(train_images)}
            )
        shutil.copyfile(src, dst)


    # Copy and split uncropped "clear" images
    for img in unc_clear_images:
        src = os.path.join(args.dataset_name, "uncropped_clear_imgs_labelled", img)
        if img in clear_train_imgs:
            dst = os.path.join(unc_train_clear_path, img)
            train_images.append({"file_name": img, "id": len(train_images)})
        else:
            dst = os.path.join(val_clear_path, img)
            val_images.append(
                {"file_name": img, "id": len(val_images) + len(train_images)}
            )
        shutil.copyfile(src, dst)



    # Copy and split cropped clear" images
    for img in crop_clear_imgs:
        src = os.path.join(args.dataset_name, "cropped_clear_imgs_manual", img)
        if img in lab_clear_train_imgs:
            dst = os.path.join(cropped_train_clear_path, img)
            train_images.append({"file_name": img, "id": len(train_images)})
        else:
            dst = os.path.join(lab_val_clear_path, img)
            val_images.append(
                {"file_name": img, "id": len(val_images) + len(train_images)}
            )
        shutil.copyfile(src, dst)


    
    # Copy and split cropped blurry" images
    for img in crop_blurry_imgs:
        src = os.path.join(args.dataset_name, "cropped_blurry_imgs_manual", img)
        if img in lab_blurry_train_imgs:
            dst = os.path.join(cropped_train_blurry_path, img)
            train_images.append({"file_name": img, "id": len(train_images)})
        else:
            dst = os.path.join(lab_val_blurry_path, img)
            val_images.append(
                {"file_name": img, "id": len(val_images) + len(train_images)}
            )
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
    num_train = len(os.listdir(good_train_folder_path)) + len(os.listdir(bad_train_folder_path))
    num_val = len(os.listdir(good_val_folder_path)) + len(os.listdir(bad_val_folder_path))
    #print(f"Number of training samples: {num_train}")
    #print(f"Number of validation samples: {(num_val)}")

    # Calculate and print the number of "good" and "bad" images in the training set
    num_good_train = len(os.listdir(good_train_folder_path))
    num_bad_train = len(os.listdir(bad_train_folder_path))
    num_good_val = len(os.listdir(good_val_folder_path))
    num_bad_val = len(os.listdir(bad_val_folder_path))
    #print(f"Number of good images in train: {num_good_train}")
    #print(f"Number of bad images in train: {num_bad_train}")
    #print(f"Number of good images in validation: {num_good_val}")
    #print(f"Number of bad images in validation: {num_bad_val}")

    # Calculate and print the number of UNCROPPED images in the training set
    unc_num_clear_train = len(os.listdir(unc_train_clear_path))
    unc_num_blurry_train = len(os.listdir(unc_train_blurry_path))
    unc_num_clear_val = len(os.listdir(val_clear_path))
    unc_num_blurry_val = len(os.listdir(val_blurry_path))

    # Calculate and print the number of CROPPED images in the training set
    cropped_num_clear_train_lab = len(os.listdir(cropped_train_clear_path))
    cropped_num_blurry_train_lab = len(os.listdir(cropped_train_blurry_path))
    cropped_num_blurry_val_lab = len(os.listdir(lab_val_blurry_path))
    cropped_num_clear_val_lab = len(os.listdir(lab_val_clear_path))

    print(f"\nNumber of uncropped clear images in train: {unc_num_clear_train}")
    print(f"Number of uncropped blurry images in train: {unc_num_blurry_train}")
    print(f"Number of uncropped clear images in validation: {unc_num_clear_val}")
    print(f"Number of uncropped blurry images in validation: {unc_num_blurry_val}\n")

    print(f"\nNumber of cropped clear LAB images in train: {cropped_num_clear_train_lab}")
    print(f"Number of cropped blurry LAB images in train: {cropped_num_blurry_train_lab}")
    print(f"Number of cropped clear LAB images in validation: {cropped_num_clear_val_lab}")
    print(f"Number of cropped blurry LAB images in validation: {cropped_num_blurry_val_lab}")


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
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Storing the split data"
    )
    args = parser.parse_args()
    split_data(args)