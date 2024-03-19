# Training and Data Preparation for Head pose estimation on MultiView videos
Implementation of dataset preparation training of 3D headpose model on DevEv videos.

## Data Preparation

1. Prepare Dataset

- [Download Corrected Headpose Dataset]()

2. Process the dataset for generating bounding box images and associated headpose for each views
```bash
python make_dataset_bodypose.py [--input_dir INPUT_DIR] [--output_dir OUTPUT_DIR] [--body_dir BODY_DIR] [--corrected_head_dir CORRECTED_HEAD_DIR] [--timestamps TIMESTAMP_FILE]
```

- `input_dir`: Path to the directory containing original videos. 
- `output_dir`: Path to the directory where the head pose dataset will be written. 
- `body_dir`: Path to the directory where body pose files are located. Default is "/nfs/hpc/cn-gpu5/DevEv/viz_bodypose/".
- `corrected_head_dir`: Directory containing attention files with corrected head orientation to copy from. (downloaded in step 1)
- `timestamps`: Path to the timestamp file. 

Example:
```bash 
python make_dataset_bodypose.py --output_dir /path/to/output/dataset/ --corrected_head_dir /path/to/corrected/attention/files/ 
```


## Training

Once the dataset is ready, it is now possible to train the headpose model. The training script [train.py](train.py) is designed to train a deep learning model for headpose estimation using a custom transformer-based architecture. Below is an explanation of the script's parameters and their default values. No modifications are necessary to use the script; however, you can adjust these parameters as needed for your specific use case.

### Usage

```bash
python train.py [options]
```

#### Example
```bash
python train.py --lr 0.0001 -n devev_results/
```

### Arguments


