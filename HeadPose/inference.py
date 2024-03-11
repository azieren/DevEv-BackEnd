import os
import argparse
import numpy as np
import cv2
import time
from PIL import Image

import torch
from torchvision import transforms
import torch.nn as nn
import torch.backends.cudnn as cudnn

from . import utils
from .model import SixDRepNet
from torchvision import transforms, models

transformations_cls = transforms.Compose([transforms.Resize((224,224)),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                      ])

transformations = transforms.Compose([transforms.Resize((224,224)),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

CURR_FOLDER = os.path.dirname(os.path.abspath(__file__))

def get_view(box, frame):
    x1, y1, x2, y2 = np.copy(box)

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

def read_track_old(path):
    data = {}

    with open(path, "r") as f:
        lines = f.readlines()

    for l in lines:
        l = l.replace("\n","")
        d = l.split(",")
        d = [eval(x) for x in d]
        
        try:
            #{},{},{},{},{},{},{},{},{},{},{
            f, _, i, x1, y1, x2, y2, _, _, h1, h2 = d
        except:
            f, i, x1, y1, x2, y2 = d
        if f not in data:
            data[f] = []
        data[f].append([x1, y1, x2, y2, i, h1, h2])  
        exit()
    return data

def read_track(path):
    data = {}

    with open(path, "r") as f:
        lines = f.readlines()
    #print(path, len(lines))
    for l in lines:
        #print(l)
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
    h, w = frame.shape[:2]
    new_bb, views = [], []
    for bb in boxes:
        body = np.copy([int(x) for x in bb[:4]])
        _, c, _ = get_view(body, frame)
        
        x_head, y_head = int(bb[5]), int(bb[6])
        x_head += w//2*(c%2)
        
        bbox_w = abs(int(bb[2]) - int(bb[0]))//3
        bbox_h = abs(int(bb[3]) - int(bb[1]))//5
        offset = max(bbox_w, bbox_h)
        
        x_min = max(x_head - offset, 0)
        y_min = max(y_head - offset, 0)
        x_max = min(w, x_head + offset)
        y_max = min(h, y_head + offset)

        if x_min >= x_max or y_min >= y_max: continue

        head = np.array([x_min, y_min, x_max, y_max], dtype=int)
        #center_x , center_y = (x_min + x_max) /2, (y_min + y_max) /2
        new_bb.append([head, body, [x_head, y_head]])
        views.append(c)
    return new_bb, views

def get_adult_classifier(checkpoint = None):
    
    model = models.resnet18(pretrained=True)
    model.fc = nn.Sequential(nn.Linear(512, 128),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(128, 2),
                                 nn.LogSoftmax(dim=1))
    if checkpoint is None:
        checkpoint = os.path.join(CURR_FOLDER, "classifier.pth")
    model = torch.load(checkpoint)
    return model

def get_front_classifier(checkpoint = None):
    
    model = models.resnet18(pretrained=True)
    model.fc = nn.Sequential(nn.Linear(512, 128),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(128, 2),
                                 nn.LogSoftmax(dim=1))
    if checkpoint is None:
        checkpoint = os.path.join(CURR_FOLDER, "frontclassifier.pth")
    model = torch.load(checkpoint)
    return model

def main_headpose(video_path, timestamps, bbox_path="/nfs/hpc/cn-gpu5/DevEv/viz_bodypose/", write = True, output_folder=None):
    cudnn.enabled = True
    gpu = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    #snapshot_path = "6DRepNet_300W_LP_AFLW2000.pth" # bad
    #snapshot_path = "6DRepNet_300W_LP_BIWI.pth"
    #snapshot_path = "6DRepNet_70_30_BIWI.pth"
    snapshot_path = os.path.join(CURR_FOLDER,"DevEv_epoch_95.pth") # finetuned
    #snapshot_path = os.path.join(CURR_FOLDER,"DevEv_epoch_30_old.pth")

    model = SixDRepNet(backbone_name='RepVGG-B1g2',
                       backbone_file='',
                       deploy=True,
                       pretrained=False)

    # Load snapshot
    saved_state_dict = torch.load(os.path.join(
        snapshot_path), map_location='cpu')

    if 'model_state_dict' in saved_state_dict:
        model.load_state_dict(saved_state_dict['model_state_dict'])
    else:
        model.load_state_dict(saved_state_dict)
    model.to(gpu)

    # Test the Model
    model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).

    ac_Classifier = get_adult_classifier()
    ac_Classifier.to(gpu)
    ac_Classifier.eval()

    fb_Classifier = get_front_classifier()
    fb_Classifier.to(gpu)
    fb_Classifier.eval()
    
    if output_folder is None:
        output_folder = os.path.join(CURR_FOLDER,'output/')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    video_name = video_path.split("/")[-1]
    video_name = video_name.split(".")[0]

    if not os.path.exists(video_path):
        print("Video not found: ", video_path)
        exit()
    video = cv2.VideoCapture(video_path)
    # New cv2
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))   # float
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)) # float
    fps = video.get(cv2.CAP_PROP_FPS)

    bbox_path = os.path.join(bbox_path, 'output_%s.txt' % video_name)
    if not os.path.exists(bbox_path):
        track = None
        print(bbox_path, "not found")
        return False
    else:
        track, type_data = read_track(bbox_path)
        
    
    # Define the codec and create VideoWriter object
    #fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    if write:
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        out = cv2.VideoWriter(os.path.join(output_folder, 'head-%s.mp4' % video_name), fourcc, fps, (width, height))

    with open(os.path.join(output_folder, 'head-%s.txt' % video_name), 'w') as f:
        f.write("")
    starts, ends = [], []
    for n, info in timestamps.items():
        for t in info:
            s, e  = t
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
    count = min(track.keys())
    end_all = max(track.keys())
    video.set(1, count)
    current_segment = 0

    while ret:         
        #if count > ends[0] + 200: break 
        if count > ends[current_segment]:
            current_segment += 1
            if current_segment < len(starts):
                count = starts[current_segment]
                video.set(1, count)
            else:
                break
        if count > end_all: break

        ret, frame = video.read()

        if track is None or not count in track:
            faces, view_list = [], []
        else:
            faces, view_list = format_bbox(track[count], frame, type_data)
            
        input_head, input_body = [], []

        for box, cam in zip(faces, view_list):
            body = box[1]
            body = frame[body[1]:body[3], body[0]:body[2]]
            body = Image.fromarray(body)
            body = body.convert('RGB')
            input_body.append(transformations_cls(body))
            
            head = box[0]
            delta = 10
            head[1], head[0] = max(0, head[1]-delta), max(0, head[0]-delta)
            head[3], head[2] = min(height, head[3]+delta), min(width, head[2]+delta)
            head = frame[head[1]:head[3], head[0]:head[2]]
            head = Image.fromarray(head)
            head = head.convert('RGB')
            head = transformations(head)
            #head = transformations(body)
            input_head.append(head)

            
        if len(faces) > 0:
            input_head = torch.stack(input_head, 0).to(gpu)
            input_body = torch.stack(input_body, 0).to(gpu)
            with torch.no_grad():
                R_pred = model(input_head)
                cls_pred = torch.softmax(ac_Classifier(input_body), dim = -1)
                cls_pred_front = torch.softmax(fb_Classifier(input_head), dim = -1)
                cls_pred_front = cls_pred_front[:, 1]
            cls_pred = assign_view(cls_pred, view_list)
            #print('Head pose estimation: %2f ms' % ((end - start)*1000.))

            euler = utils.compute_euler_angles_from_rotation_matrices(
                R_pred)*180/np.pi
            y_pred_deg = euler[:, 0].cpu()
            p_pred_deg = euler[:, 1].cpu()
            r_pred_deg = euler[:, 2].cpu()       

            for box_info, p, y, r, label, front_label in zip(faces, p_pred_deg, y_pred_deg, r_pred_deg, cls_pred, cls_pred_front):
                if label == 0: c = (0,255,0)
                else: c = (0,0,255)
                head = box_info[0]
                body = box_info[1]
                head_width = abs(head[2] - head[0])*0.8
                if write:
                    """utils.plot_pose_cube(frame,  y, p, r, int(head[0] + .5*(
                        head[2]-head[0])), int(head[1] + .5*(head[3]-head[1])), size=int(head_width))"""
                    utils.draw_axis(frame, y, p, r, int(head[0] + .5*(
                        head[2]-head[0])), int(head[1] + .5*(head[3]-head[1])), size=int(head_width))               
                    cv2.rectangle(frame, (head[0], head[1]),  (head[2], head[3]), c, 2)
                    cv2.rectangle(frame, (body[0], body[1]),  (body[2], body[3]), c, 1)
                with open(os.path.join(output_folder, 'head-%s.txt' % video_name), 'a') as f:
                    f.write("{},{},{:.2f},{},{},{},{},{:.3f},{:.3f},{:.3f}\n".format(count, label, front_label, 
                                head[0], head[1], head[2], head[3], y.item(), p.item(), r.item()))
                      
        if write: out.write(frame)
        count += 1
        if count % 100 == 0:
            print("Segment {}/{} - frame {} - Start frame {} - End frame {}".format(current_segment+1, len(starts), 
                    count, starts[current_segment], ends[current_segment]))
        if count > starts[0] + 1000: break

    out.release()
    video.release()

    return True

if __name__ == '__main__':
    video_path = ""
    bbox_path = ""
    main_headpose(video_path, bbox_path, write = True)
