import os
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from PIL import Image
import numpy as np
import torch
from torch import nn
from torchvision import transforms, models

transformations = transforms.Compose([transforms.Resize((224,224)),
                                      transforms.ToTensor(),
                                      ])

def get_model():
    device = torch.device("cuda" if torch.cuda.is_available() 
                                  else "cpu")
    
    
    model = models.resnet18(pretrained=True)
    model.fc = nn.Sequential(nn.Linear(512, 128),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(128, 2),
                                 nn.LogSoftmax(dim=1))

    return model, device


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

def format_bbox(boxes, frame, type_data):

    exclude_b = []

    new_bb = []
    for bb in boxes:
        #print(bb)
        p = bb[4]
        if type_data == 0:
            
            x_min = int(bb[0])
            y_min = int(bb[1])
            x_max = int(bb[2])
            y_max = int(bb[3])   
        
        else:
            bbox_w = abs( int(bb[2]) - int(bb[0]))//5
            bbox_h = abs(int(bb[1]) - int(bb[3]))//6
            
            x_min = max(int(bb[5]) - bbox_w, 0)
            y_min = max(int(bb[6]) - bbox_h, 0)
            x_max = min(frame.shape[1], int(bb[5]) + bbox_w)
            y_max = min(frame.shape[0], int(bb[6]) + bbox_h)

            x_min = max(int(bb[5]) - bbox_w, 0)
            y_min = max(int(bb[6]) - bbox_w, 0)
            x_max = min(frame.shape[1], int(bb[5]) + bbox_w)
            y_max = min(frame.shape[0], int(bb[6]) + bbox_w)

            if x_min >= x_max or y_min >= y_max: continue
            #x_min -= 5
            #x_max += 5
            #y_min -= 5
            #y_max += 5
            x_min = max(x_min, 0)
            y_min = max(y_min, 0)
            x_max = min(frame.shape[1], x_max)
            y_max = min(frame.shape[0], y_max)

        center_x , center_y = (x_min + x_max) /2, (y_min + y_max) /2
        new_bb.append([x_min, y_min, x_max, y_max, p])
    return new_bb


def assign_view(preds, views):
    preds = preds.cpu().numpy()
    views = np.array(views)
    subview = np.unique(views)
    
    labels = np.zeros(len(preds))
    
    for v in subview:
        p = preds[views == v]
        if len(p) == 1:
            l = np.argmax(p, axis = -1)
            labels[views == v] = l
        else:
            temp = torch.zeros(len(p))
            l = np.argmax(p[:,1], axis = 0)
            temp[l] = 1 ## Check most likely child
            labels[views == v] = temp

    return labels

def inference(video_path, bbox_path, output_folder, checkpath='output/childclassifier_best.pth', write = True):
    model, device = get_model()
    model = torch.load(checkpath)
    model.to(device)
    model.eval()

    video_name = video_path.split("/")[-1]
    video_name = video_name.split(".")[0]

    video = cv2.VideoCapture(video_path)
    # New cv2
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))   # float
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)) # float
    fps = video.get(cv2.CAP_PROP_FPS)

    # Define the codec and create VideoWriter object
    #fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    if write:
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        out = cv2.VideoWriter(os.path.join(output_folder, 'output-%s.mp4' % video_name), fourcc, fps, (width, height))


    if not os.path.exists(bbox_path):
        track = None
        print(bbox_path, "not found")
        return False
    else:
        track, type_data = read_track(bbox_path)

    ret = True
    starts = min(track.keys())
    ends = max(track.keys())
    count = starts + 5000
    video.set(1, count)
    

    while ret:         
        if count > starts + 5200: break 
        if count > ends+1: break 

        ret, frame = video.read()

        if track is None or not count in track:
            faces = []
        else:
            faces = track[count]

        img_list, box_list, view_list = [], [], []
        for box in faces:
            x_min = int(box[0])
            y_min = int(box[1])
            x_max = int(box[2])
            y_max = int(box[3])

            img = frame[y_min:y_max, x_min:x_max]
            _, c, _ = get_view(box[:4], frame)

            img = Image.fromarray(img)
            img = img.convert('RGB')
            img_list.append(transformations(img))
            box_list.append([x_min, y_min, x_max, y_max])
            view_list.append(c)

        img_list = torch.stack(img_list, 0).to(device)
        with torch.no_grad():
            pred = torch.softmax(model(img_list), dim = -1)
        pred = assign_view(pred, view_list)

        for p, box in zip(pred, box_list):
            if p == 0: c = (0,255,0)
            else: c = (0,0,255)
            cv2.rectangle(frame, (box[0], box[1]),  (box[2], box[3]), c, 3)

        if write: 
            out.write(frame)
        count += 1
        if count % 100 == 0:
            print("Process: ", count)

    out.release()
    video.release()

    return True
    
def get_view(box, frame):
    x1, y1, x2, y2 = box

    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)

    h, w = frame.shape[:2]
    h, w = [h//4, w//2]
    if cx <= w and cy <= h:
        c_ = 0 
        new_frame = frame[:h, :w]
        new_box = [x1, y1, x2, y2]
    elif cx > w and cy <= h:
        new_frame = frame[:h, w:]
        c_ = 1
        new_box = [x1-w, y1, x2-w, y2]
    elif cx <= w and 2*h >= cy > h:
        new_frame = frame[h:2*h, :w]
        c_ = 2
        new_box = [x1, y1-h, x2, y2-h]
    elif cx > w and 2*h >= cy > h:
        c_ = 3      
        new_box = [x1-w, y1-h, x2-w, y2-h]
        new_frame = frame[h:2*h, w:]
    elif cx <= w and 3*h >= cy > 2*h:
        c_ = 4      
        new_box = [x1, y1-2*h, x2, y2-2*h]
        new_frame = frame[2*h:3*h, :w]
    elif cx > w and 3*h >= cy > 2*h:
        c_ = 5  
        new_box = [x1-w, y1-2*h, x2-w, y2-2*h]
        new_frame = frame[2*h:3*h, w:]
    elif cx <= w and cy > 3*h:
        c_ = 6  
        new_box = [x1, y1-3*h, x2, y2-3*h]
        new_frame = frame[3*h:, :w]
    else:
        c_ = 7
        new_box = [x1-w, y1-3*h, x2-w, y2-3*h]
        new_frame = frame[3*h:, w:]
    return new_box, c_, new_frame

def bbox_iou(box1, box2):
    """
    Computes IoU between two bounding boxes

    Args:
        box1: A list of 4 integers representing the bounding box (x1, y1, x2, y2)
        box2: A list of 4 integers representing the bounding box (x1, y1, x2, y2)

    Returns:
        iou: A float representing the Intersection over Union (IoU) between the two bounding boxes
    """
    # Calculate intersection area
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)

    # Calculate union area
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    union_area = box1_area + box2_area - intersection_area

    # Calculate IoU
    iou = intersection_area / union_area

    return iou


def assign_ids(boxes_prev, boxes_curr, ids_prev, iou_threshold=0.3):
    # Number of boxes in the previous and current frame
    n_prev, n_curr = len(boxes_prev), len(boxes_curr)
    
    # Distance matrix between previous and current frame boxes
    dist_mat = np.zeros((n_prev, n_curr))
    for i in range(n_prev):
        for j in range(n_curr):
            iou = bbox_iou(boxes_prev[i], boxes_curr[j])
            if iou >= iou_threshold:
                dist_mat[i, j] = 1.0 - iou
    
    # Assign ids to the current frame boxes based on distance
    ids_curr = np.zeros(n_curr)
    used_ids = set()
    for j in range(n_curr):
        # Find the previous frame box with the minimum distance
        if len(boxes_prev) > 0:
            i = np.argmin(dist_mat[:, j])
            dist = dist_mat[i, j]
        else:
            # No boxes in previous frame, assign new id
            i, dist = -1, np.inf
        
        # Assign id to the current frame box
        if dist < np.inf and ids_prev[i] not in used_ids:
            ids_curr[j] = ids_prev[i]
            used_ids.add(ids_prev[i])
        else:
            # Assign new id to the current frame box
            if len(used_ids) > 0:
                new_id = max(used_ids) + 1
            else:
                new_id = 1
            ids_curr[j] = new_id
            used_ids.add(new_id)
    
    return ids_curr

def inference_naive(video_path, bbox_path, output_folder,write = True):

    video_name = video_path.split("/")[-1]
    video_name = video_name.split(".")[0]

    video = cv2.VideoCapture(video_path)
    # New cv2
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))   # float
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)) # float
    fps = video.get(cv2.CAP_PROP_FPS)

    # Define the codec and create VideoWriter object
    #fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    if write:
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        out = cv2.VideoWriter(os.path.join(output_folder, 'output2-%s.mp4' % video_name), fourcc, fps, (width, height))


    if not os.path.exists(bbox_path):
        track = None
        print(bbox_path, "not found")
        return False
    else:
        track, type_data = read_track(bbox_path)

    ret = True
    starts = min(track.keys())
    ends = max(track.keys())
    count = starts + 5000
    video.set(1, count)
    
    bbox_prev = {i:[] for i in range(8)}

    threshold = 30
    ids_prev_dict = {i:[0,1] for i in range(8)}
    while ret:         
        if count > starts + 5200: break 
        if count > ends+1: break 

        ret, frame = video.read()

        if track is None or not count in track:
            faces = []
        else:
            faces = track[count]


        box_list = {i:[] for i in range(8)}
        box_list_or = {i:[] for i in range(8)}
        for i, box in enumerate(faces):
            x_min = int(box[0])
            y_min = int(box[1])
            x_max = int(box[2])
            y_max = int(box[3])

            bbox, c, _ = get_view(box[:4], frame)
            box_list[c].append(bbox)
            box_list_or[c].append(box)


        for c in range(8):
            ids = assign_ids(bbox_prev[c], box_list[c], ids_prev_dict[c])
            ids =  [int(x) for x in ids ]
            for i, id in enumerate(ids):
                box = [int(x) for x in box_list_or[c][i] ]
                if id == 0: color = (0,255,0)
                else: color = (0,0,255)
                cv2.rectangle(frame,  (box[0], box[1]),  (box[2], box[3]), color=color, thickness=2)  
                cv2.putText(frame, str(id), (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            bbox_prev[c] = box_list[c]          

        if write: 
            out.write(frame)
        count += 1
        if count % 100 == 0:
            print("Process: ", count)

    out.release()
    video.release()

    return True

if __name__ == "__main__":

    output_dir = "output/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    video_path = "/nfs/hpc/cn-gpu5/DevEv/dataset/DevEv_S13_05_Sync.mp4"
    bbox_path = "/nfs/hpc/cn-gpu5/DevEv/viz_bodypose/output_DevEv_S13_05_Sync.txt"
    inference(video_path, bbox_path, output_dir)
    #inference_naive(video_path, bbox_path, output_dir)
    