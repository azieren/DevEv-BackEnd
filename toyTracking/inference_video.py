import numpy as np
import cv2
import torch
import glob as glob
import os
import time
import argparse
import yaml
import matplotlib.pyplot as plt

import torchvision
from torchvision import transforms as transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from deep_sort_realtime.deepsort_tracker import DeepSort

def convert_pre_track(
    draw_boxes, pred_classes, scores
):
    final_preds = []
    for i, box in enumerate(draw_boxes):
        # Append ([x, y, w, h], score, label_string). For deep sort real-time.
        final_preds.append(
            (
                [box[0], box[1], box[2] - box[0], box[3] - box[1]],
                scores[i],
                str(pred_classes[i])
            )
        )
    return final_preds

def convert_post_track(
    tracks
):
    draw_boxes, pred_classes, scores, track_id = [], [], [], []
    for track in tracks:
        if not track.is_confirmed():
            continue
        score = track.det_conf
        if score is None:
            continue
        track_id = track.track_id
        pred_class = track.det_class
        pred_classes.append(f"{track_id} {pred_class}")
        scores.append(score)
        draw_boxes.append(track.to_ltrb())
    return draw_boxes, pred_classes, scores

def create_model(num_classes, pretrained=True, coco_model=False):
    # Load Faster RCNN pre-trained model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
        weights='DEFAULT'
    )
    if coco_model: # Return the COCO pretrained model for COCO classes.
        return model, coco_model
    
    # Get the number of input features 
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # define a new head for the detector with required number of classes
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes) 

    return model

def infer_transforms(image):
    # Define the torchvision image transforms.
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ])
    return transform(image)

def resize(im, img_size=640, square=False):
    # Aspect ratio resize
    if square:
        im = cv2.resize(im, (img_size, img_size))
    else:
        h0, w0 = im.shape[:2]  # orig hw
        r = img_size / max(h0, w0)  # ratio
        if r != 1:  # if sizes are not equal
            im = cv2.resize(im, (int(w0 * r), int(h0 * r)))
    return im

def read_return_video_data(video_path):
    cap = cv2.VideoCapture(video_path)
    # Get the video's frame width and height
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    assert (frame_width != 0 and frame_height !=0), 'Please check video path...'
    return cap, frame_width, frame_height

def select_frame_per_view(frame, view):
    h, w = frame.shape[:2]
    h, w = [h//4, w//2]
    if view == 0:
        new_frame = frame[:h, :w]
    elif view == 1:
        new_frame = frame[:h, w:]
    elif view == 2:
        new_frame = frame[h:2*h, :w]
    elif view == 3:
        new_frame = frame[h:2*h, w:]
    elif view == 4:
        new_frame = frame[2*h:3*h, :w]
    elif view == 5:
        new_frame = frame[2*h:3*h, w:]
    elif view == 6:
        new_frame = frame[3*h:, :w]
    else:
        new_frame = frame[3*h:, w:]
    return new_frame

def get_box_from_view(box, view, h, w):
    x1, y1, x2, y2 = box
    return np.array([x1 + w*(view%2), y1 + h*(view//2), x2 + w*(view%2), y2 + h*(view//2)])

def get_model(path, device, num_classes):
    checkpoint = torch.load(path, map_location=device)
    model = create_model(num_classes=num_classes, coco_model=False)
    new_state_dict = {}
    for key, value in checkpoint['model_state_dict'].items():
        if key.startswith('module.'):  # Remove 'module' prefix
            new_state_dict[key[7:]] = value
        else:
            new_state_dict[key] = value
    
    #model.load_state_dict(checkpoint['model_state_dict'])
    model.load_state_dict(new_state_dict)
    model.to(device).eval() 
    return model   

def inference_annotations(
    draw_boxes, 
    pred_classes, 
    scores, 
    classes,
    colors, 
    orig_image, 
):
    lw = 3 # max(round(sum(orig_image.shape) / 2 * 0.003), 2)  # Line width.
    tf = max(lw - 1, 1) # Font thickness.
    
    # Draw the bounding boxes and write the class name on top of it.
    for j, box in enumerate(draw_boxes):
        p1 = (int(box[0]), int(box[1])) #(int(box[0]/image.shape[1]*width), int(box[1]/image.shape[0]*height))
        p2 = (int(box[2]), int(box[3])) #(int(box[2]/image.shape[1]*width), int(box[3]/image.shape[0]*height))
        class_name = pred_classes[j]
        color = colors[classes.index(' '.join(class_name.split(' ')[1:]))]

        cv2.rectangle(
            orig_image,
            p1, p2,
            color=color, 
            thickness=lw,
            lineType=cv2.LINE_AA
        )
        # For filled rectangle.
        final_label = class_name + ' ' + str(round(scores[j], 2))
        w, h = cv2.getTextSize(
            final_label, 
            cv2.FONT_HERSHEY_SIMPLEX, 
            fontScale=lw / 3, 
            thickness=tf
        )[0]  # text width, height
        w = int(w - (0.20 * w))
        outside = p1[1] - h >= 3
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        cv2.rectangle(
            orig_image, 
            p1, 
            p2, 
            color=color, 
            thickness=-1, 
            lineType=cv2.LINE_AA
        )  
        cv2.putText(
            orig_image, 
            final_label, 
            (p1[0], p1[1] - 5 if outside else p1[1] + h + 2),
            cv2.FONT_HERSHEY_SIMPLEX, 
            fontScale=lw / 3.8, 
            color=(255, 255, 255), 
            thickness=tf, 
            lineType=cv2.LINE_AA
        )
    return orig_image

def main_inference_toy(video_output, timestamps, write=True, output_folder=''):
    # For same annotation colors each time.
    np.random.seed(42)
    # Define the detection threshold any detection having
    # score below this will be discarded.
    detection_threshold = 0.75
    
    tracker = {}    
    for i in range(8):
        tracker[i] = DeepSort(max_age=30)

    CURR_DIR = os.path.dirname(os.path.realpath(__file__))
    # Load the data configurations.
    data_configs = CURR_DIR + "/data_configs/devev.yaml"
    with open(data_configs) as file:
        data_configs = yaml.safe_load(file)
    NUM_CLASSES = data_configs['NC']
    CLASSES = data_configs['CLASSES']
        
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    OUT_DIR = output_folder
    VIDEO_PATH = video_output

    # Load weights if path provided.
    model = get_model(CURR_DIR + "/outputs/training/res_3/best_model.pth", DEVICE, NUM_CLASSES)

    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
    assert VIDEO_PATH is not None, 'Please provide path to an input video...'

    cap, frame_width, frame_height = read_return_video_data(VIDEO_PATH)   
    fps = cap.get(cv2.CAP_PROP_FPS)
    save_name = VIDEO_PATH.split(os.path.sep)[-1].split('.')[0]
    # Define codec and create VideoWriter object.
    if write:
        out = cv2.VideoWriter(f"{OUT_DIR}/{save_name}_toys.mp4", 
                        cv2.VideoWriter_fourcc(*'mp4v'), fps, 
                        (frame_width, frame_height))
    np.save(f"{OUT_DIR}/{save_name}_toys.npy", {})
    
    starts, ends = [], []
    for n, info in timestamps.items():
        for (s, e) in info:
            starts.append(int(fps*s/1000))
            ends.append(int(fps*e/1000))

    if len(timestamps) == 0:
        starts, ends = [0], [np.inf]
    else:
        index = np.argsort(starts)
        starts, ends = np.array(starts), np.array(ends)
        starts = starts[index]
        ends = ends[index]

    ret = True
    frame_count = starts[0]
    end_all = ends[-1]
    cap.set(1, frame_count)
    current_segment = 0     

     # To count total frames.
    frame_height, frame_width = frame_height//4, frame_width//2
    RESIZE_TO = frame_width
    
    data_output = {}
    # read until end of video
    while ret:
        if frame_count > ends[current_segment]:
            current_segment += 1
            if current_segment < len(starts):
                frame_count = starts[current_segment]
                cap.set(1, frame_count)
            else:
                break
        if frame_count > end_all: break
        # capture each frame of the video
        ret, frame = cap.read()
        if not ret: break

        
        image_tensor, frame_list = [], []
        for i in range(8):
            frame_v = select_frame_per_view(frame, i)
            frame_v = resize(frame_v, RESIZE_TO, square=False)
            frame_list.append(frame_v.copy())
            frame_v = cv2.cvtColor(frame_v, cv2.COLOR_BGR2RGB)
            frame_v = infer_transforms(frame_v)
            image_tensor.append(frame_v)
        # Add batch dimension.
        image_tensor = torch.stack(image_tensor, 0)
        with torch.no_grad():
            # Get predictions for the current frame.
            outputs = model(image_tensor.to(DEVICE))

        # Load all detection to CPU for further operations.
        outputs = [{k: v.cpu().numpy() for k, v in t.items()} for t in outputs]
        
        frame_data = {}
        # Carry further only if there are detected boxes.
        for i in range(8):
            boxes = outputs[i]['boxes']
            scores = outputs[i]['scores']
            labels = outputs[i]['labels']
            if len(boxes) == 0: continue
            
            boxes = boxes[scores >= detection_threshold].astype(np.int32)
            if len(boxes) == 0: continue
            labels = labels[scores >= detection_threshold]
            scores = scores[scores >= detection_threshold]
            pred_classes = [CLASSES[i] for i in labels]

            # Get all the predicited class names.
            tracker_inputs = convert_pre_track(
                boxes, pred_classes, scores
            )
            # Update tracker with detections.
            tracks = tracker[i].update_tracks(tracker_inputs, frame=frame_list[i])
            draw_boxes, pred_classes, scores = convert_post_track(tracks) 

            track_ids = [track.track_id for track in tracks if track.is_confirmed() and track.det_conf is not None] 
            frame_data[i] = {"labels":labels, "label_names":pred_classes, 
                             "bboxes":np.copy(draw_boxes), "scores":scores, "track_ids":track_ids}

            if write: 
                draw_boxes = [get_box_from_view(box, i, frame_height, frame_width) for box in draw_boxes]
                frame = inference_annotations(draw_boxes, 
                                            pred_classes, 
                                            scores,
                                            CLASSES, 
                                            COLORS, 
                                            frame)
        #break
        
        if write: out.write(frame)
        if frame_count % 100 == 0: 
            print("Segment {} - Frame {}/{}".format(current_segment, frame_count, ends[current_segment] - starts[current_segment]))
        # Increment frame count.
        data_output[frame_count] = frame_data
        frame_count += 1          
        if frame_count > starts[0] + 3000: break  
        
    np.save(f"{OUT_DIR}/{save_name}_toys.npy", data_output)
    # Release VideoCapture().
    cap.release()
    if write: out.release()
    return

if __name__ == '__main__':
    main_inference_toy()