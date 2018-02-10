# Tumory
Liver lesions detector

## HoGs
### Dataset
#### Prepare your dataset
##### Create dataset manually
1. Insert positive frames into -> dataset/processed/orig_images.\n
2. Insert negative frames into -> dataset/processed/negatives.\n
3. Insert test frames into -> dataset/processed/test_images.\n

Images should be in format .png or .pklz.

##### OR
1. Insert CT data (.pklz format) to **CTs/** \n
2. Extract slices from CT data (.pklz format) \n
> python data_augmentation.py
3. Copy dataset into target folders. \n
> python file_manager.py


## CNN
