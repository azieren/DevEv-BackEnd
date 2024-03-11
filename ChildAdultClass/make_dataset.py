import os
from collections import OrderedDict
import numpy as np 
import cv2
import re

def get_processed_body(path = "BodyPose/output"):
    # soft = True  reprocess unfinished files
    # soft = False remove fully processed files
    end = ".npz"
    return [ f for f in os.listdir(path) if f.endswith(end) ]

def get_downloaded_videos(path_vid):
    record = OrderedDict()
    for f in os.listdir(path_vid):
        if not f.endswith("_Sync.mp4"): continue
        sess_name = re.findall(r'\d\d_\d\d', f)[0]
        record[sess_name] = os.path.join(path_vid, f)

    return record

def get_body_video(path_vid, path_body):
    process = get_processed_body(path_body)
    videos = get_downloaded_videos(path_vid)

    new_videos = OrderedDict()
    for p in process:
        sess_name = re.findall(r'\d\d_\d\d', p)[0]
        if sess_name in videos:
            boxfile = p.replace("data_2d_", "output_").replace(".npz", ".txt")
            new_videos[sess_name] = {"video":videos[sess_name] , "bbox": os.path.join(path_body, boxfile) }
    return new_videos

def read_track(path):
    data = {}

    with open(path, "r") as f:
        lines = f.readlines()

    for l in lines:
        l = l.replace("\n","")
        d = l.split(",")
        d = [eval(x) for x in d]
        
        if len(d) == 9:
            f, p, x1, y1, x2, y2, _, h1, h2 = d
            type_data = 0
        elif len(d) == 10:
            f, p, x1, y1, x2, y2, _, _, h1, h2 = d
            type_data = 1
        else:
            f, _, p, x1, y1, x2, y2, _, _, h1, h2 = d
            type_data = 1
            

        if f not in data:
            data[f] = []
        data[f].append([x1, y1, x2, y2, p, h1, h2])  

    return data, type_data


def extract_bb(name, video_path, bbox_path, output_dir, N= 30):
    if not os.path.exists(bbox_path):
        track = None
        print(bbox_path, "not found")
        return False
    else:
        track, type_data = read_track(bbox_path)

    frames = list(track.keys())
    frames = sorted(np.random.choice(frames, N))
    
    video = cv2.VideoCapture(video_path)
    # New cv2
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))   # float
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)) # float
    fps = video.get(cv2.CAP_PROP_FPS)
    

    for t in frames:
        video.set(1, t)
        ret, image = video.read()
        if not ret: 
            print("Error frame ", t)
            continue
        bboxes = track[t]

        for i, bb in enumerate(bboxes):
            x_min, y_min = int(bb[0]), int(bb[1])
            x_max, y_max = int(bb[2]), int(bb[3])  
            img = image[y_min:y_max, x_min:x_max]
            bb_name = os.path.join(output_dir, "{}_{}_{}.png".format(name, t, i))
            cv2.imwrite(bb_name, img)
    return

if __name__ == "__main__":
    video_dir = "/nfs/hpc/cn-gpu5/DevEv/dataset/" # output dir to save videos
    path_processed = "/nfs/hpc/cn-gpu5/DevEv/viz_bodypose/"
    output_dir = "output_bbox2/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get processed videos
    video_dict = get_body_video(video_dir, path_processed)
    count = 1
    for name, info in video_dict.items():
        #if count < 70: 
        #    count += 1
        #    continue
        # Select only first half of available files
        extract_bb(name, info["video"], info["bbox"], output_dir, N = 5)
        print("Finished {} - {}/{}".format(name, count, len(video_dict)))
        count += 1
