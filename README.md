# DevEv - Developping Environemnt - Training and inference codebase
Implementation of body, head pose, and attention estimation of infant, projection of head pose and toys in 3D, toy tracking by detection, and child/adult classifier.

This code is the official repository for the ICDL 2024 [publication]():
`Infants’ Developing Environment: Integration of Computer Vision and Human Annotation to Quantify Where Infants Go, What They Touch, and What They See ` from Han et al.


## Installation

1. Create and Activate Environment then Clone the Repository:

```bash
conda create -n devev_env
conda activate devev_env
git clone https://github.com/azieren/DevEv-BackEnd.git
```

2. Navigate to the repository directory and install the required dependencies using pip:

```bash
cd DevEv-BackEnd
pip install -r requirements.txt
```

3. Download Pretrained Models
   
Download the pretrained models from the following links:

- [Download BodyPose Model](https://drive.google.com/file/d/1r_pjyqLFOP8sO1dYC_zZSUqJ5PBZ4LLm/view?usp=drive_link)
- [Download HeadPose Model for Room View](https://drive.google.com/file/d/1QMIPOBYdQwJ9HF4tJQbB1INtmkOEelsj/view?usp=drive_link)
- [Download HeadPose Model for Mat view](https://drive.google.com/file/d/1WBDN4M5CsZlrarTvoS8VIjNknPltgyp9/view?usp=drive_link)
- [Download Child Adult Classifier](https://drive.google.com/file/d/1r_pjyqLFOP8sO1dYC_zZSUqJ5PBZ4LLm/view?usp=drive_link)
- [Download Toy Tracking Model](https://example.com/toy_tracking_model.pth)

4. Place Pretrained Models

Place the downloaded pretrained models in the following directories of this repository.

- Place BodyPose Model `infant_w48_384x288.pth` in `DevEv-BackEnd/BodyPose/`
- Place HeadPose Model for Room View `DevEvMV.pth` in `DevEv-BackEnd/HeadPose/`
- Place HeadPose Model for Mat view `DevEvMatMV.pth` in `DevEv-BackEnd/HeadPose/`
- Place Child Adult Classifier `childClassifier.pth` in `DevEv-BackEnd/HeadPose/`
- Place Toy Tracking Model `toyModel.pth` in `DevEv-BackEnd/toyTracking/`

## Inference Pipeline

1. Infant Attention Pipeline

- `main_bodypose.py`: Process the video to get the body bounding box and keypoints of all person in the 8 views
- `main_headpose.py`: Process the video to track the infant and infer the headpose in 3D
- `main_2D3D_mv.py`: Project the head and hands and attention in 3D
- `main_3Dcone_collision.py`: Model the attention as a cone projected in 3D to capture collision between attention and room/toys

2. Toy Pipeline

- `main_toy_track.py`: Track the toys in 2D
- `main_2D3D_toys.py`: Project the centroid of toys from 2D to 3D

3. Other

- `calibration.py`: Used for generating camera parameters for calibration
- `main_eval.py`: Used for evaluating error between 2 set of files containing data on the same sessions
- `make_dataset_bodypose`: Used for training the headpose model (see [here](HeadPose/README.md))

## Recurring Arguments for Inference scripts

These arguments are commonly used across multiple scripts:
- `--input_dir INPUT_DIR`: Path to the folder containing raw video files (if a video is missing it will be automatically downloaded from databrary)
- `--output_dir OUTPUT_DIR`: Path to folder where write results
- `--timestamps TIMESTAMP_FILE`: Path to the timestamp file, set by default to "DevEvData_2023-06-20.csv".
- `--uname USERNAME`: Databrary username.
- `--psswd PASSWORD`: Databrary password.
- `--write`: If set, a video will be generated alongside the corresponding output file with visualization of the type of processed data.
- `--check_time`: Used only for checking the current amount of frames processed by existing files.
- `--check_remaining`: Used only for checking the files that have not been processed yet.
- `--session`: Used only for processing a single video, use the format ##_## (session and subject number) to specify which video to process.


## Body Detection and Keypoints: main_bodypose.py

This script performs body pose estimation on videos and generates body pose files. Optionally, it can also generate visualization videos alongside the pose files. If the original does not exist in the input folder, then it will be automatically downloaded from databrary.

### Usage

```bash
python main_bodypose.py [--input_dir INPUT_DIR] [--output_dir OUTPUT_DIR] [--timestamps TIMESTAMP_FILE] [--uname USERNAME] [--psswd PASSWORD] [--write] [--check_time] [--check_remaining] [--session SESSION]
```

#### Example
```bash
python main_bodypose.py --input_dir /path/to/videos/ --output_dir /path/to/output/
```

### Output

For each video, two files will be written:

1. **NPZ File**: Contains body pose information with the following format:
    - `dataset`: A dictionnary where a key is a frame number and values are 2D detection information
    - `dataset[frame] = {"views": views, "bbox": bbox_tensor, "kpt": kps}`
      - `frame`: Frame number.
      - `views`: List of views ids in the video with detection for a particular frame.
      - `bbox`: List of Associated bounding boxes to each view.
      - `kpt`: List of keypoints in each bounding box.

2. **TXT File**: Contains body pose information in text format where each rows have the following information: 
(frame, 0, b_id, x_min, y_min, x_max, y_max, x_l, y_l, x_n, y_n)
    - `frame`: Frame number.
    - `b_id`: Body ID.
    - `x_min`, `y_min`, `x_max`, `y_max`: Bounding box coordinates.
    - `x_l`, `y_l`: 2D Position for keypoint "left ankle".
    - `x_n`, `y_n`: 2D Position for keypoint "nose" .

## Head Pose in 3D: main_head.py

This script performs child/adult classification to detect and track the child in the video for every view, as well as predicting the head orientation in 3D. It takes the bounding box and keypoint information output from "main_bodypose.py" as input, along with the original videos. 

While the main script only does inference, it is also possible to retrain a new model. The data preparation can be achieved by using a dataset composed of corrected attention file and processed video then running [make_dataset_bodypose.py](make_dataset_bodypose.py) for processing the data and prepare them for training the headpose model. The training code can be found in [here](HeadPose/README.md)

### Usage

```bash
python main_head.py [--input_dir INPUT_DIR] [--output_dir OUTPUT_DIR] [--body_dir BODY_DIR] [--timestamps TIMESTAMP_FILE] [--uname USERNAME] [--psswd PASSWORD] [--write] [--check_time] [--check_remaining] [--session SESSION]
```

#### Example
```bash
python main_head.py --input_dir /path/to/videos/ --output_dir /path/to/output/ --body_dir /path/to/bodypose/
```

### Output
For each video, a .txt file will be written  where each rows have the following information: 
(frame, label, x_min, y_min, x_max, y_max,, R_x, R_y, R_z)

1. **TXT File**: For each video, a .txt file will be written where each row has the following format:
    - `frame`: Frame number.
    - `label`: Adult/Child label.
    - `x_min`, `y_min`, `x_max`, `y_max`: Head bounding box coordinate
    - `R_x`, `R_y`, `R_z`: Head orientation as a 3D vector associated to the head bounding box


## Head and Hands in 3D: main_2D3D_mv.py

This script takes as input the output from `main_headpose.py` and projects the head location, hands location from 2D to 3D, and head orientation and attention points. It uses the 3D model of the room to compute collision between the attention and the room for computing the attention point.

### Usage

```bash
python main_2D3D_mv.py [--output_dir OUTPUT_DIR] [--body_dir BODY_DIR] [--head_dir HEAD_DIR] [--timestamps TIMESTAMP_FILE] [--uname USERNAME] [--psswd PASSWORD] [--check_time] [--video_dir VIDEO_DIR] [--session SESSION]
```

#### Example
```bash
python main_2D3D_mv.py --output_dir /path/to/output/ --body_dir /path/to/bodypose/ --head_dir /path/to/headpose/
```

### Output
1. **TXT File**: For each video, a .txt file will be written where each row has the following format: 
(frame, flag_a, flag_h, head_x, head_y, head_z, att_x, att_y, att_z, handL_x, handL_y, handL_z, handR_x, handR_y, handR_z)
    - `frame`: Frame number.
    - `flag_a`=0: Placeholder flag telling whether the attention/head position has been corrected in the frame (set to zero)
    - `flag_h`=0: Placeholder flag telling whether a hand position has been corrected in the frame (set to zero)
    - `head_x`, `head_y`, `head_z`: Projected head location in 3D. (x,y,z)
    - `att_x`, `att_y`, `att_z`: Projected attention point location in 3D. (x,y,z)
    - `handL_x`, `handL_y`, `handL_z`: Projected left hand location in 3D. (x,y,z)
    - `handR_x`, `handR_y`, `handR_z`: Projected right hand location in 3D. (x,y,z)

## Toy Tracking in 2D: main_toy_track.py

This script performs tracking of toys in videos. The instruction for dataset preparation and training can be found in [toyTracking/README.md](toyTracking/README.md)

### Usage

```bash
python main_toy_track.py [--input_dir INPUT_DIR] [--output_dir OUTPUT_DIR] [--timestamps TIMESTAMP_FILE] [--uname USERNAME] [--psswd PASSWORD] [--write] [--session SESSION]
```

#### Example
```bash
python main_toy_track.py --input_dir /path/to/videos/ --output_dir /path/to/output/
```

### Output
1. **NPY File**: For each video, a .npy file containing a dictionary will be written to the specified output directory. The dictionary has the following format:

```python
data[frame][view_id] = {
    "labels": labels,
    "label_names": pred_classes,
    "bboxes": np.copy(draw_boxes),
    "scores": scores,
    "track_ids": track_ids
}
```
  - `frame`: Frame number.
  - `view_id`: View id
  - `label_names`: List of detected objects by name
  - `bboxes`: List of bounding boxes of detected objects
  - `scores`: List of confidence score associated to each bounding box
  - `track_ids`: List of track id associated to each bounding box

## Toy Tracking in 3D: main_2D3D_toys.py

This script projects each detected and tracked toys from 2D to 3D.

### Usage

```bash
python main_2D3D_toys.py [--toy_dir TOY_DIR] [--output_dir OUTPUT_DIR] [--timestamps TIMESTAMP_FILE] [--uname USERNAME] [--psswd PASSWORD] [--session SESSION]
```

#### Example
```bash
python main_2D3D_toys.py --input_dir /path/to/toy_tracking_data/ --output_dir /path/to/output/
```
### Output

1. **NPY File**: For each toy, a .npy file will be written to the specified output directory. The file has the following format:

```python
toy_data[toy_name][frame] = {
    "p3d": p3d,
    "track": track_list
}
```
  - `toy_name`: Name of the toy considered.
  - `frame`: Frame number.
  - `p3d`: 3D location of the toy in a frame
  - `track`: list of tracklets for this toy

## Cone and moving room collision in 3D: main_3Dcone_collision.py

This script takes as input the output from `main_2D3D_mv.py` and `main_2D3D_toys.py`, and projects the lines belonging to a cone into the room while updating toys locations. It uses the 3D model of the room to compute collision between the attention and the room for computing the attention point and the toy tracking results to update position of toys.

### Usage

```bash
python main_3Dcone_collision.py [--output_dir OUTPUT_DIR] [--att_dir ATT_DIR] [--toy_dir TOY_DIR] [--cone_angle CONE_ANGLE] [--n_lines N_LINES] [--session SESSION]
```

- `cone_angle`: Angle of the cone in degree (from 0 to 90 degrees)
- `n_lines`: Number of lines to sample uniformly in the cone
- `att_dir`: Directory where attention files are saved
- `toy_dir`: Directory where toy 3D centroid locations files are saved

#### Example
```bash
python main_3Dcone_collision.py --output_dir /path/to/output/ --att_dir /path/to/attention/ --toy_dir /path/to/3Dtoy/ --cone_angle 40 --n_lines 100
```

### Output
1. **TXT File**: For each video, a .txt file will be written where each row has the following format: 
(frame, line id, object name, att_x, att_y, att_z, head_x, head_y, head_z, )
    - `frame`: Frame number.
    - `line id`: Line id that was sample from the cone and projected into the room
    - `object name`: Object name the line collided with
    - `att_x`, `att_y`, `att_z`: Projected attention point location in 3D. (x,y,z) the line collided with
    - `head_x`, `head_y`, `head_z`: Projected head location in 3D. (x,y,z)
