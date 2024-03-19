# toyTracking
Implementation of dataset preparation training and inference of faster-rcnn detection model for 2D object detection and tracking on DevEv videos.

## Data Preparation

1. Prepare Dataset


- [Download Toy Dataset](https://drive.google.com/file/d/1r_pjyqLFOP8sO1dYC_zZSUqJ5PBZ4LLm/view?usp=drive_link)

2. Process the dataset for generating single view images and ground-truth from multiple view dataset
```bash
python extract_frames.py [--input_img_dir INPUT_IMG_DIR] [--input_gt_dir INPUT_GT_DIR] [--output_img_dir OUTPUT_IMG_DIR] [--output_gt_dir OUTPUT_GT_DIR]
```
--`input_img_dir`: Path to multiview image folder. Default is "DatasetDetector/images_raw/".
--`input_gt_dir`: Path to multiview ground truth folder. Default is "DatasetDetector/gt_raw/".
--`output_img_dir`: Path to single-view image folder to write dataset. Default is "DatasetDetector/gt/".
--`output_gt_dir`: Path to single-view ground truth folder to write dataset. Default is "DatasetDetector/img/"


Example:
```bash 
python extract_frames.py --input_img_dir /path/to/multiview/images --input_gt_dir /path/to/multiview/gt --output_img_dir /path/to/singleview/images --output_gt_dir /path/to/singleview/gt
```

3. Place Dataset
Place the downloaded and processed dataset in a folder of your choice then modify accordingly the information in [data_configs/devev.yaml](data_configs/devev.yaml). Reset the path, the set of classes and the number of classes according to the dataset you are using.
- `TRAIN_DIR_IMAGES`: Path to single view images for the training set from `output_img_dir`
- `TRAIN_DIR_LABELS`: Path to single view annotations (.json) for the training set from `output_gt_dir`
- `VALID_DIR_IMAGES`: Path to single view images for the test set (any validation set you setup or the same as `TRAIN_DIR_IMAGES`)
- `VALID_DIR_LABELS`: Path to single view annotations (.json) for the test set (any validation set you setup or the same as `TRAIN_DIR_IMAGES`)

## Training

Once the dataset is ready, it is now possible to train the detector. The training script [train.py](train.py) is designed to train a deep learning model for object detection using the Faster R-CNN architecture. Below is an explanation of the script's parameters and their default values. No modifications are necessary to use the script; however, you can adjust these parameters as needed for your specific use case.

### Usage

```bash
python train.py [options]
```

#### Example
```bash
python train.py --lr 0.0001 -n devev_results/
```

### Arguments
- m, --model: Name of the model architecture. Default is fasterrcnn_resnet50_fpn_v2.
- data: Path to the data configuration file. Default is data_configs/devev.yaml.
- d, --device: Computation/training device. Default is cuda, which uses the GPU if available.
- e, --epochs: Number of epochs to train for. Default is 20.
- j, --workers: Number of workers for data processing/transforms/augmentations. Default is 4.
- b, --batch: Batch size to load the data. Default is 4.
- lr: Learning rate for the optimizer. Default is 0.001.
- ims, --imgsz: Image size to feed to the network. Default is 640.
- n, --name: Training result directory name in outputs/training/. Default is res_#.
- vt, --vis-transformed: Visualize transformed images fed to the network. Default is False.
- mosaic: Probability of applying mosaic. Default is 0.0, which always applies mosaic.
- uta, --use-train-aug: Whether to use train augmentation. Default is False.
- ca, --cosine-annealing: Use cosine annealing warm restarts. Default is False.
- w, --weights: Path to model weights if using pretrained weights. Default is None.
- r, --resume-training: Whether to resume training. Default is False.
- st, --square-training: Resize images to square shape instead of aspect ratio resizing for single image training. Default is False.
- world-size: Number of distributed processes. Default is 1.
- dist-url: URL used to set up the distributed training. Default is env://.
- dw, --disable-wandb: Whether to disable Weights & Biases logging. Default is False.
- sync-bn: Use synchronized batch normalization. Default is False.
- amp: Use automatic mixed precision. Default is False.
- seed: Global seed for training. Default is 0.
- project-dir: Save results to a custom directory instead of the outputs directory. Default is None.

