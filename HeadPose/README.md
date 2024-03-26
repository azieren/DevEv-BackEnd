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
python train_head_pose.py [--gpu GPU_ID] [--num_epochs NUM_EPOCHS] [--batch_size BATCH_SIZE] [--lr LEARNING_RATE]
                          [--dataset DATASET_TYPE] [--data_dir DATA_DIRECTORY] [--filename_list FILENAME_LIST]
                          [--output_string OUTPUT_STRING] [--setup SETUP_TYPE] [--timestamps TIMESTAMP_FILE]
                          [--snapshot SNAPSHOT_PATH]
```

#### Example
```bash
python train_head_pose.py --gpu 0 --num_epochs 50 --batch_size 64 --lr 0.00008 --dataset DevEvMV --data_dir /path/to/headpose/dataset/img/ --filename_list /path/to/gt_headpose.txt --output_string Eval_DevEvMat --setup mat --timestamps /path/to/timestamps.csv 
```

### Arguments
- `gpu`: GPU device ID to use (default: 0).
- `num_epochs`: Maximum number of training epochs (default: 50).
- `batch_size`: Batch size for training (default: 64).
- `lr`: Base learning rate for the optimizer (default: 0.00008).
- `dataset`: Type of dataset to use (default: 'DevEvMV').
- `data_dir`: Directory path containing the dataset with images (default: '/nfs/hpc/cn-gpu5/DevEv/headpose_dataset/bodyhead_dataset_new/').
- `filename_list`: Path to a text file containing path to ground truth head poses (default: '/nfs/hpc/cn-gpu5/DevEv/headpose_dataset/gt_body_new.txt').
- `output_string`: String appended to output snapshots during training (default: 'Eval_DevEvMat').
- `setup`: Type of camera setup ('room' or 'mat') (default: 'room').
- `timestamps`: Path to the timestamps file (default: "DevEvData_2024-02-02.csv").
- `snapshot`: Path of the model snapshot to initialize the bodypose model in training (default: '../BodyPose/infant_w48_384x288.pth').

