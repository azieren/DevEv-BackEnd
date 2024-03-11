import os
from collections import OrderedDict
import numpy as np 
import cv2
import re
import json

def get_timestamp(filename="DevEvData_2023-04-26.csv"):
    with open(filename) as f:
        text = f.readlines()
    
    text = [l.split(",") for l in text[1:]]
    record = OrderedDict()
    for data in text:
        if data[1] not in record:
            # Processed flag: False means the the method has not been processed yet
            record[data[1]] = {"head":False, "body":False}
        if len(data) <= 25: category = data[-3]
        else: category = data[-6]
        if category in ['c', 'r', 'p']:
            if len(data) <= 25:
                onset, offset = int(data[-2]), int(data[-1])
            else:
                onset, offset = int(data[-5]), int(data[-4])
            if category not in record[data[1]]: record[data[1]][category] = []
            record[data[1]][category].append((onset, offset))
            
    return record

def get_downloaded_videos(path_vid):
    timestamps = get_timestamp("/nfs/hpc/share/azieren/DevEv/DevEvData_2023-06-20.csv")
    record = OrderedDict()
    for f in os.listdir(path_vid):
        if not f.endswith("_Sync.mp4"): continue
        sess_name = re.findall(r'\d\d_\d\d', f)[0]
        if not sess_name in timestamps:
            print("Error Timestamp not found", sess_name)
            continue
        record[sess_name] = {"path":os.path.join(path_vid, f), "timestamp":timestamps[sess_name] }

    return record

def extract_frames(name, video_info, output_dir, N= 30):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    timestamps = video_info["timestamp"]   
    timestamps = {key: timestamps[key] for key in ['c', 'r', 'p'] if key in timestamps}
    video_path = video_info["path"]    
    video = cv2.VideoCapture(video_path)
    # New cv2
    fps = video.get(cv2.CAP_PROP_FPS)
    
    # Initialize Timestamps
    frames = []
    for n, info in timestamps.items():
        for t in info:
            s, e  = t
            starts = int(fps*s/1000)
            ends = int(fps*e/1000)
            frame_list = np.arange(starts, ends)
            frames.extend(np.random.choice(frame_list, N))

    frames = sorted(frames)
    for t in frames:
        video.set(1, t)
        ret, image = video.read()
        if not ret: 
            print("Error frame ", t)
            continue
        output_name = os.path.join(output_dir, "{}_{}.png".format(t, name))
        cv2.imwrite(output_name, image)
    return

def extract_all(video_dir, output_dir):
    # Get processed videos
    video_dict = get_downloaded_videos(video_dir)
    count = 1
    for name, info in video_dict.items():
        #if count < 70: 
        #    count += 1
        #    continue
        # Select only first half of available files
        extract_frames(name, info, output_dir, N = 2)
        print("Finished {} - {}/{}".format(name, count, len(video_dict)))
        count += 1
    return

def get_img_view(img, view):
    img = cv2.imread(img)
    h, w = img.shape[:2]
    h, w = [h//4, w//2]
    if view == 0:
        new_frame = img[:h, :w]
    elif view == 1:
        new_frame = img[:h, w:]
    elif view == 2:
        new_frame = img[h:2*h, :w]
    elif view == 3:
        new_frame = img[h:2*h, w:]
    elif view == 4:
        new_frame = img[2*h:3*h, :w]
    elif view == 5:
        new_frame = img[2*h:3*h, w:]
    elif view == 6:
        new_frame = img[3*h:, :w]
    else:
        new_frame = img[3*h:, w:]
    return new_frame


def get_gt_view(gt, view, img):
    h, w = img.shape[:2]
    with open(gt, "rb") as f:
        gt_data = json.load(f)

    new_data = []
    for label in gt_data['shapes']:
        b0, b1 = label["points"]
        x1, y1, x2, y2 = b0[0], b0[1], b1[0], b1[1]
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)

        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        if cx <= w and cy <= h:
            c_ = 0 
            new_box = [x1, y1, x2, y2]
        elif cx > w and cy <= h:
            c_ = 1
            new_box = [x1-w, y1, x2-w, y2]
        elif cx <= w and 2*h >= cy > h:
            c_ = 2
            new_box = [x1, y1-h, x2, y2-h]
        elif cx > w and 2*h >= cy > h:
            c_ = 3      
            new_box = [x1-w, y1-h, x2-w, y2-h]
        elif cx <= w and 3*h >= cy > 2*h:
            c_ = 4      
            new_box = [x1, y1-2*h, x2, y2-2*h]
        elif cx > w and 3*h >= cy > 2*h:
            c_ = 5  
            new_box = [x1-w, y1-2*h, x2-w, y2-2*h]
        elif cx <= w and cy > 3*h:
            c_ = 6  
            new_box = [x1, y1-3*h, x2, y2-3*h]
        else:
            c_ = 7
            new_box = [x1-w, y1-3*h, x2-w, y2-3*h]
        if not c_ == view: continue
        x0, x1 = max(new_box[0], 0), min(new_box[2], w)
        y0, y1 = max(new_box[1], 0), min(new_box[3], h)
        if x0 > x1:
            print(view, label["label"], x0, x1)
            print(gt)
            exit()
        bbox = list(map(int, [x0, y0, x1, y1]))
        new_data.append({"label":label["label"], "bbox":bbox})
    return new_data

def process_gt(img_dir, gt_dir, output_gt_dir="gt/", output_img_dir="img/"):
    img_list = [x for x in os.listdir(img_dir) if os.path.exists(os.path.join(gt_dir, x.replace(".png", ".json")))]

    if not os.path.exists(output_gt_dir):
        os.makedirs(output_gt_dir)
    if not os.path.exists(output_img_dir):
        os.makedirs(output_img_dir)
        
    for img in img_list:
        img_path = os.path.join(img_dir, img)
        annotations = os.path.join(gt_dir, img.replace(".png", ".json"))
        for i in range(8):
            view = get_img_view(img_path, i)
            gt = get_gt_view(annotations, i, view)
            
            im_name = "{}_{}".format(i, img)
            cv2.imwrite(os.path.join(output_img_dir, im_name), view)
            gt_name = "{}_{}".format(i, img.replace(".png", ".json"))
            json.dump(gt, open(os.path.join(output_gt_dir, gt_name), "w"))
    
    return

if __name__ == "__main__":
    video_dir = "/nfs/hpc/cn-gpu5/DevEv/dataset/" # output dir to save videos
    output_dir = "DatasetDetector/images_raw/"
    gt_dir = "DatasetDetector/gt_raw/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    #extract_all(video_dir, output_dir)
    process_gt(output_dir, gt_dir, output_gt_dir="DatasetDetector/gt/", output_img_dir="DatasetDetector/img/")



