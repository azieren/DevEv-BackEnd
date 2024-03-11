import logging
import os
import pandas as pd
import cv2
import numpy as np
import json

from torch.utils.tensorboard.writer import SummaryWriter



logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def set_log(log_dir):
    logging.basicConfig(
        # level=logging.DEBUG,
        format='%(message)s',
        # datefmt='%a, %d %b %Y %H:%M:%S',
        filename=f"{log_dir}/train.log",
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # add the handler to the root logger
    logging.getLogger().addHandler(console)

def log(content, *args):
    for arg in args:
        content += str(arg)
    logger.info(content)

def coco_log(log_dir, stats):
    log_dict_keys = [
        'Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]',
        'Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ]',
        'Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ]',
        'Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ]',
        'Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]',
        'Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ]',
        'Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ]',
        'Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ]',
        'Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]',
        'Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ]',
        'Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]',
        'Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ]',
    ]
    log_dict = {}
    # for i, key in enumerate(log_dict_keys):
    #     log_dict[key] = stats[i]

    with open(f"{log_dir}/train.log", 'a+') as f:
        f.writelines('\n')
        for i, key in enumerate(log_dict_keys):
            out_str = f"{key} = {stats[i]}"
            logger.debug(out_str) # DEBUG model so as not to print on console.
        logger.debug('\n'*2) # DEBUG model so as not to print on console.
    # f.close()

def set_summary_writer(log_dir):
    writer = SummaryWriter(log_dir=log_dir)
    return writer

def tensorboard_loss_log(name, loss_np_arr, writer, epoch):
    """
    To plot graphs for TensorBoard log. The save directory for this
    is the same as the training result save directory.
    """
    writer.add_scalar(name, loss_np_arr[-1], epoch)

def tensorboard_map_log(name, val_map_05, val_map, writer, epoch):
    writer.add_scalars(
        name,
        {
            'mAP@0.5': val_map_05[-1], 
            'mAP@0.5_0.95': val_map[-1]
        },
        epoch
    )

def create_log_csv(log_dir):
    cols = [
        'epoch', 
        'map', 
        'map_05',
        'train loss',
        'train cls loss',
        'train box reg loss',
        'train obj loss',
        'train rpn loss'
    ]
    results_csv = pd.DataFrame(columns=cols)
    results_csv.to_csv(os.path.join(log_dir, 'results.csv'), index=False)

def csv_log(
    log_dir, 
    stats, 
    epoch,
    train_loss_list,
    loss_cls_list,
    loss_box_reg_list,
    loss_objectness_list,
    loss_rpn_list
):
    if epoch+1 == 1:
        create_log_csv(log_dir) 
    
    df = pd.DataFrame(
        {
            'epoch': int(epoch+1),
            'map_05': [float(stats[0])],
            'map': [float(stats[1])],
            'train loss': train_loss_list[-1],
            'train cls loss': loss_cls_list[-1],
            'train box reg loss': loss_box_reg_list[-1],
            'train obj loss': loss_objectness_list[-1],
            'train rpn loss': loss_rpn_list[-1]
        }
    )
    df.to_csv(
        os.path.join(log_dir, 'results.csv'), 
        mode='a', 
        index=False, 
        header=False
    )

def overlay_on_canvas(bg, image):
    bg_copy = bg.copy()
    h, w = bg.shape[:2]
    h1, w1 = image.shape[:2]
    # Center of canvas (background).
    cx, cy = (h - h1) // 2, (w - w1) // 2
    bg_copy[cy:cy + h1, cx:cx + w1] = image
    return bg_copy * 255.


def append_annotation_to_coco(output, image_info, output_filename):
    """
    Log outputs to a JSON file in COCO format during inference.

    :param output: Output from the model.
    :param image_info: Dictionary containing file name, width, and heigh.
    :param output_filename: Path to the JSON file.
    """
    if not os.path.exists(output_filename):
        # Initialize file with basic structure if it doesn't exist
        with open(output_filename, 'w') as file:
            json.dump({"images": [], "annotations": [], "categories": []}, file, indent=4)

    with open(output_filename, 'r') as file:
        coco_data = json.load(file)

    annotations = coco_data['annotations']
    images = coco_data['images']
    categories = set(cat['id'] for cat in coco_data['categories'])
    annotation_id = max([ann['id'] for ann in annotations], default=0) + 1
    image_id = len(images) + 1

    # Add image entry
    images.append({
        "id": image_id, 
        "file_name": image_info['file_name'], 
        "width": image_info['width'], 
        "height": image_info['height']
    })

    boxes = output['boxes'].tolist()
    labels = output['labels'].tolist()

    for box, label in zip(boxes, labels):
        xmin, ymin, xmax, ymax = box
        width = xmax - xmin
        height = ymax - ymin

        annotation = {
            "id": annotation_id,
            "image_id": image_id,
            "bbox": [xmin, ymin, width, height],
            "area": width * height,
            "category_id": label,
            "iscrowd": 0
        }
        annotations.append(annotation)
        annotation_id += 1
        categories.add(label)

    # Update categories
    coco_data['categories'] = [{"id": cat_id, "name": f"category_{cat_id}"} for cat_id in categories]

    with open(output_filename, 'w') as file:
        json.dump(coco_data, file, indent=4)

def log_to_json(image, out_path, outputs):
    """
    Function to call when saving JSON log file during inference.
    No other file should need calling other than this to keep the code clean.

    :param image: The original image/frame.
    :param out_path: Path where the JSOn file should be saved.
    :param outputs: Model outputs.
    """
    image_info = {
        "file_name": "frame1.jpg", "width": image.shape[1], "height": image.shape[0]
    }
    append_annotation_to_coco(outputs[0], image_info, out_path)