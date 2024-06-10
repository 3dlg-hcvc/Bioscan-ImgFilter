## 1. Complete, processed coco annotations: 

```
python scripts/complete_coco_json.py --input_dir data/failed_crop_subset
```

## 2. Split data into train and val:

```
python scripts/split_data.py --input_dir data/failed_crop_subset --dataset_name data/data_splits
```

## 3. To view visualizations of good/bad train images, loss, and accuracy scores:

```
python scripts/visualizations.py --goodTrain_folder_path ./data/data_splits/train/good --badTrain_folder_path ./data/data_splits/train/bad
```

## 4. To train the model:
```
python scripts/training.py
```


## 5. To evaluate the model:
```
python scripts/training.py
```
# Image-Filtering-Tool
Detrmines whether an image has a valid or invalid bounding box 
