# Image-Filtering-Tool (need to improve this)
Detrmines whether an image has a valid or invalid bounding box 

## 1. Complete, processed coco annotations: 

```
python data_processing/complete_coco_json.py --input_dir data/failed_crop_subset
```

## 2. Split data into train and val:

```
python data_processing/split_data.py --input_dir data/failed_crop_subset --dataset_name data/data_splits
```

## 3. To view visualizations of good/bad train images, loss, and accuracy scores:

```
python Visualizations/visualizations.py --goodTrain_folder_path ./data/data_splits/train/good --badTrain_folder_path ./data/data_splits/train/bad
```

## 4. To train the model:
```
python scripts/training.py
```

## 5. To view model inference:
```
python scripts/inference.py
```
