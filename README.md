# DevEv-BackEnd
Implementation of body and head pose estimation of infant, projection of head pose and toys in 3D, toy tracking by detection, and child/adult classifier

## Installation

1. Clone the repository:

```bash
git clone https://github.com/azieren/DevEv-BackEnd.git
cd DevEv-BackEnd
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

## Recurring Arguments for Inference scripts

These arguments are commonly used across multiple scripts:

- `--timestamps TIMESTAMP_FILE`: Path to the timestamp file, it set by default to "DevEvData_2023-06-20.csv".
- `--uname USERNAME`: Databrary username.
- `--psswd PASSWORD`: Databrary password.
- `--write`: If set, a video will be generated alongside the corresponding output file with visualization of the type of processed data.
- `--check_time`: Used only for checking the current amount of frames processed by existing files.
- `--check_remaining`: Used only for checking the files that have not been processed yet.

## main_bodypose.py

This script performs body pose estimation on videos and generates body pose files. Optionally, it can also generate visualization videos alongside the pose files. If the original does not exist in the input folder, then it will be automatically downloaded from databrary.

### Usage

```bash
python main_bodypose.py [--input_dir INPUT_DIR] [--output_dir OUTPUT_DIR] [--timestamps TIMESTAMP_FILE] [--uname USERNAME] [--psswd PASSWORD] [--write] [--check_time] [--check_remaining]
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

## main_head.py

This script performs child/adult classification to detect and track the child in the video for every view, as well as predicting the head orientation in 3D. It takes the bounding box and keypoint information output from "main_bodypose.py" as input, along with the original videos.

### Usage

```bash
python main_head.py [--input_dir INPUT_DIR] [--output_dir OUTPUT_DIR] [--body_dir BODY_DIR] [--timestamps TIMESTAMP_FILE] [--uname USERNAME] [--psswd PASSWORD] [--write] [--check_time] [--check_remaining]
```

#### Example
```bash
python main_head.py --input_dir /path/to/videos/ --output_dir /path/to/output/ --body_dir /path/to/bodypose/
```

### Output
For each video, a .txt file will be written  where each rows have the following information: 
(count, label, x_min, y_min, x_max, y_max,, R_x, R_y, R_z)

1. **TXT File**: For each video, a .txt file will be written with the following format:
    - `frame`: Frame number.
    - `label`: Adult/Child label.
    - `x_min`, `y_min`, `x_max`, `y_max`: Head bounding box coordinate
    - `R_x`, `R_y`, `R_z`: Head orientation as a 3D vector associated to the head bounding box


## main_2D3D_mv.py

This script takes as input the output from `main_headpose.py` and projects the head location, hands location from 2D to 3D, and head orientation and attention points. It uses the 3D model of the room to compute collision between the attention and the room for computing the attention point.

### Usage

```bash
python main_2D3D_mv.py [--output_dir OUTPUT_DIR] [--body_dir BODY_DIR] [--head_dir HEAD_DIR] [--timestamps TIMESTAMP_FILE] [--uname USERNAME] [--psswd PASSWORD] [--check_time] [--video_dir VIDEO_DIR]
```

#### Example
```bash
python main_2D3D_mv.py --output_dir /path/to/output/ --body_dir /path/to/bodypose/ --head_dir /path/to/headpose/
```

### Output
1. **TXT File**: For each video, a .txt file will be written with the following format: 
(frame, flag_a, flag_h, head_x, head_y, head_z, att_x, att_y, att_z, handL_x, handL_y, handL_z, handR_x, handR_y, handR_z)
    - `frame`: Frame number.
    - `flag_a`=0: Placeholder flag telling whether the attention/head position has been corrected in the frame (set to zero)
    - `flag_h`=0: Placeholder flag telling whether a hand position has been corrected in the frame (set to zero)
    - `head_x`, `head_y`, `head_z`: Projected head location in 3D. (x,y,z)
    - `att_x`, `att_y`, `att_z`: Projected attention point location in 3D. (x,y,z)
    - `handL_x`, `handL_y`, `handL_z`: Projected left hand location in 3D. (x,y,z)
    - `handR_x`, `handR_y`, `handR_z`: Projected right hand location in 3D. (x,y,z)

## main_toy_track.py

This script performs tracking of toys in videos. The instruction for dataset preparation and training can be found in [toyTracking/README.md](toyTracking/README.md)

### Usage

```bash
python main_toy_track.py [--input_dir INPUT_DIR] [--output_dir OUTPUT_DIR] [--timestamps TIMESTAMP_FILE] [--uname USERNAME] [--psswd PASSWORD] [--write]
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

## main_2D3D_toys.py

This script projects each detected and tracked toys into 3D.

### Usage

```bash
python main_2D3D_toys.py [--input_dir INPUT_DIR] [--output_dir OUTPUT_DIR] [--timestamps TIMESTAMP_FILE] [--uname USERNAME] [--psswd PASSWORD]
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
  - `track`: list of availbale tracklet for this toy