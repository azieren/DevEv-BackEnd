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
from scipy.spatial.transform import Rotation  
from torchvision import transforms, models

from . import utils
from .MVPose import MVPoseNet
from .posenet_model import get_pose_net

transformations = transforms.Compose([transforms.Resize((224,224)),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

CURR_FOLDER = os.path.dirname(os.path.abspath(__file__))

PARENT_BOX_B_1 = [685, 418, 960, 540]

def get_view(box, frame):
    x1, y1, x2, y2 = np.copy(box)

    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)

    h, w = frame.shape[:2]
    h, w = h//4, w//2
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

def assign_view(preds, views, cam_type, body):
    preds = torch.softmax(preds, dim=-1).cpu().numpy()
    views = np.array(views)
    subview = np.unique(views)
    
    labels = np.zeros(len(preds), dtype=int)
    
    # 0: Adult, 1: Background, 2:Children
    for v in subview:
        p = preds[views == v]
        #print(v, p)
        #if v in [0,1,2,3] and cam_type == "c":
        # Case of only one detection in view
        if len(p) == 1:
            l = np.argmax(p, axis = -1)
            labels[views == v] = l
            #if v in [0,1,2,3] and cam_type == "c":
            #    labels[views == v] = 2
        else: # Case of multiple detection in view
            temp = torch.zeros(len(p))
            ## Check most likely child
            l = np.argmax(p[:,1], axis = 0)
            temp[l] = 1
            # If it is an obvious adult: score of child class smaller than threshold
            if p[l,1] < 0.05:
                temp[l] = 0 
            labels[views == v] = temp
    return labels

def read_bodyposes(bodyfile):
    bodyfile = bodyfile.replace("output_", "data_2d_").replace(".txt", ".npz")
    if not os.path.exists(bodyfile):
        print(bodyfile, "not found")
        exit()
    bodyinfo = np.load(bodyfile, allow_pickle=True)
    bodyinfo = bodyinfo["data"].item()

    data = {}
    for f, body in bodyinfo.items():
        body_kpt = np.array(body["kpt"], dtype=int).astype(float)
        body_bbox = np.array(body["bbox"], dtype=int).astype(float)
        
        frame = int(f)
        if frame not in data:
            data[frame] = []
        for kpt, bbox in zip(body_kpt, body_bbox):
            #0: 'nose',
            #1: 'left_eye',
            #2: 'right_eye',
            left_eye = kpt[1,:]
            right_eye = kpt[2,:]
            head = (left_eye+right_eye)/2.0
            head = list(map(int, head))
            
            box = list(map(int, bbox[:4]))
            data[frame].append([box[0], box[1], box[2], box[3], 1, head[0], head[1]])

    return data, 1

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
        elif len(d) == 11:
            f, _, p, x1, y1, x2, y2, _, _, h1, h2 = d
            type_data = 1
        else: continue
            
        f = int(f)
        if f not in data:
            data[f] = []
        data[f].append([x1, y1, x2, y2, p, h1, h2])  

    return data, type_data

def format_bbox(boxes, frame, type_data, is_mat=False):

    exclude_b = []
    h, w = frame.shape[:2]
    new_bb, views = [], []
    for bb in boxes:
        body = np.copy([int(x) for x in bb[:4]])
        _, c, _ = get_view(body, frame)
        if abs(body[1] - body[3]) < 35 or abs(body[0] - body[2]) < 35 : continue
        
        x_head, y_head = int(bb[5]), int(bb[6])
        #print(c, x_head, y_head, y_head % (h//4), h//4, w//2)
        # Remove boundaries
        if abs(x_head % (w//2) - (w//2)) <= 30 : continue
        if abs(y_head % (h//4) - (h//4) ) <= 30: continue
        # Remove impossible cases
        if c in [5] and (y_head% (h//4) <= 90) : continue
        if c in [5] and (PARENT_BOX_B_1[0]  <= x_head % (w//2) <= PARENT_BOX_B_1[2] and PARENT_BOX_B_1[1] <= y_head% (h//4) <= PARENT_BOX_B_1[3]) : continue
        # Remove impossible cases
        if c in [4] and (y_head% (h//4) <= 50) : continue
                
        # Check only mat views
        if is_mat and c not in [0,1,2,3,6]: continue
        if is_mat and c in [0,1,2,3]:
            if abs(x_head - (w//2)) <= 50 : continue
            if abs(y_head % (h//4) - (h//4) ) <= 50: continue
        
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
    # 0/1: Adult/Child

    model = models.resnet50(pretrained=True)
    model.fc = nn.Sequential(nn.Linear(2048,2))
    
    
    """model = models.resnet18(pretrained=True)
    model.fc = nn.Sequential(nn.Linear(512, 256),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 #nn.Linear(128, 2),
                                 nn.Linear(256, 2),
                                 nn.LogSoftmax(dim=1))

    model = models.resnet34(pretrained=True)
    model.fc = nn.Sequential(nn.Linear(512,128),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(128, 2),
                                 nn.LogSoftmax(dim=1))   """
    if checkpoint is None:
        checkpoint = os.path.join(CURR_FOLDER, "childClassifier.pth")
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

def prep_features(features, views, ac_label):
    _, D = features.size()
    
    children_view = views[ac_label == 1]
    #print(views, ac_label, children_view, features.size())
    multiview_features = torch.zeros((1,8,D)).float().to(features.device)
    multiview_features[0, children_view] = features[ac_label == 1]
    
    view_onehot = torch.zeros((1,8)).long().to(features.device)
    view_onehot[0, children_view] = 1
    return multiview_features, view_onehot

def load_mv_model(model_R, typeview, room, mat):
    model_R = model_R.cpu()
    if typeview != "c":
        saved_state_dict = torch.load(os.path.join(
            room), map_location='cpu')
    else:
        saved_state_dict = torch.load(os.path.join(
            mat), map_location='cpu')
                
    if 'model_R_state_dict' in saved_state_dict:
        model_R.load_state_dict(saved_state_dict['model_R_state_dict'])
    else:
        model_R.load_state_dict(saved_state_dict)
        
    return model_R

def main_headpose(video_path, timestamps, bbox_path="/nfs/hpc/cn-gpu5/DevEv/viz_bodypose/", write = True, output_folder=None):
    cudnn.enabled = True
    gpu = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    #snapshot_path = "6DRepNet_300W_LP_AFLW2000.pth" # bad
    #snapshot_path = "6DRepNet_300W_LP_BIWI.pth"
    snapshot_path_backbone = 'BodyPose/infant_w48_384x288.pth'
    snapshot_path = os.path.join(CURR_FOLDER,"DevEvMV.pth") # finetuned
    snapshot_path_mat = os.path.join(CURR_FOLDER,"DevEvMatMV.pth") # finetuned
    #snapshot_path = os.path.join(CURR_FOLDER,"DevEv_epoch_30_old.pth")

    #model = SixDRepNet(backbone_name='RepVGG-B1g2',
    #                   backbone_file='',
    #                   deploy=True,
    #                   pretrained=False)
    model = get_pose_net(is_train=False)


    model_R = MVPoseNet(feature_dim=384+17*2)

    # Load snapshot
    saved_state_dict = torch.load(os.path.join(
        snapshot_path_backbone), map_location='cpu')

    if 'model_state_dict' in saved_state_dict:
        model.load_state_dict(saved_state_dict['model_state_dict'])
    else:
        model.load_state_dict(saved_state_dict)
        

    model.to(gpu)
    model_R.to(gpu)

    # Test the Model
    model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
    model_R.eval()
    
    ac_Classifier = get_adult_classifier()
    ac_Classifier.to(gpu)
    ac_Classifier.eval()

    #fb_Classifier = get_front_classifier()
    #fb_Classifier.to(gpu)
    #fb_Classifier.eval()
    
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
        #track, type_data = read_bodyposes(bbox_path)
        
    
    # Define the codec and create VideoWriter object
    #fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    if write:
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        out = cv2.VideoWriter(os.path.join(output_folder, 'head-%s.mp4' % video_name), fourcc, fps, (width, height))

    cam_mat = np.load("camera_mat.npy", allow_pickle = True).item()
    cam_room = np.load("camera_room.npy", allow_pickle = True).item()


    with open(os.path.join(output_folder, 'head-%s.txt' % video_name), 'w') as f:
        f.write("")
    starts, ends, cams = [], [], []
    for n, info in timestamps.items():
        for t in info:
            s, e  = t
            starts.append(int(fps*s/1000))
            ends.append(int(fps*e/1000))
            cams.append(n)

    if len(timestamps) == 0:
        starts, ends, cams = [0], [np.inf], ["p"]
    else:
        index = np.argsort(starts)
        starts, ends, cams = np.array(starts), np.array(ends), np.array(cams)
        starts = starts[index]
        ends = ends[index]
        cams = cams[index]
        
    
    ret = True
    count = min(track.keys()) 
    end_all = max(track.keys())
    video.set(1, count)
    current_segment = 0
    curr_cam = cam_room
    if cams[current_segment] == 'c':
        curr_cam = cam_mat
    model_R = load_mv_model(model_R, cams[current_segment], snapshot_path, snapshot_path_mat).to(gpu)
    
    R = np.array([0.0,0.0,1.0])
    while ret:         
        if count > ends[current_segment]:
            current_segment += 1
            if current_segment < len(starts):
                count = starts[current_segment]
                video.set(1, count)
                curr_cam = cam_room
                if cams[current_segment] == 'c':
                    curr_cam = cam_mat      
                model_R = load_mv_model(model_R, cams[current_segment], snapshot_path, snapshot_path_mat).to(gpu)
                R = np.array([0.0,0.0,1.0])
            else:
                break
        if count > end_all: break

        ret, frame = video.read()

        if track is None or not count in track:
            faces, view_list = [], []
        else:
            faces, view_list = format_bbox(track[count], frame, type_data, is_mat=cams[current_segment] == 'c')
        view_list = np.array(view_list)    
        input_body = []

        #bb = 0
        for box in faces:
            body = box[1] 
            #delta = 10
            #body[1], body[0] = max(0, body[1]-delta), max(0, body[0]-delta)
            #body[3], body[2] = min(height, body[3]+delta), min(width, body[2]+delta)
            body = frame[body[1]:body[3], body[0]:body[2]]
            #cv2.imwrite("{}.png".format(bb), body)
            #bb += 1
            body = cv2.cvtColor(body, cv2.COLOR_BGR2RGB)
            body = Image.fromarray(body)
            body = body.convert('RGB')
            input_body.append(transformations(body))
                        
        if len(faces) > 0:
            #input_head = torch.stack(input_head, 0).to(gpu)
            input_body = torch.stack(input_body, 0).to(gpu)
            with torch.no_grad():
                feature = model.forward_feature(input_body)
                
                cls_pred = ac_Classifier(input_body)
                #print(cls_pred)
                #exit()
                cls_pred = assign_view(cls_pred, view_list, cams[current_segment], body)
                feature, view_1hot = prep_features(feature, view_list, cls_pred)   
                if view_1hot.sum() > 0:
                    R = model_R(feature, view_1hot).squeeze().cpu().numpy()
            
            #print('Head pose estimation: %2f ms' % ((end - start)*1000.))
            attention = project_att(R, view_list, cls_pred, curr_cam)     
            y_pred_deg, p_pred_deg, r_pred_deg = attention[:,0], attention[:,1], attention[:,2]
                        
            # Adult
            for box_info, y, p, r, label in zip(faces, y_pred_deg, p_pred_deg, r_pred_deg, cls_pred):
                if label == 0: c = (0,255,0) # Adult green
                #elif label == 1: c = (255,255,255) # BG green
                else: c = (255,0,0)
                head = box_info[0]
                body = box_info[1]
                head_width = abs(head[2] - head[0])*0.8
                if write:
                    """utils.plot_pose_cube(frame,  y, p, r, int(head[0] + .5*(
                        head[2]-head[0])), int(head[1] + .5*(head[3]-head[1])), size=int(head_width))"""
                    utils.draw_axis(frame, y, p, r, int(head[0] + .5*(
                        head[2]-head[0])), int(head[1] + .5*(head[3]-head[1])), size=int(head_width))               
                    #cv2.rectangle(frame, (head[0], head[1]),  (head[2], head[3]), c, 2)
                    cv2.rectangle(frame, (body[0], body[1]),  (body[2], body[3]), c, 3)
                with open(os.path.join(output_folder, 'head-%s.txt' % video_name), 'a') as f:
                    f.write("{},{},{},{},{},{},{:.3f},{:.3f},{:.3f}\n".format(count, label, 
                                head[0], head[1], head[2], head[3], R[0], R[1], R[2]))
                      
        if write: out.write(frame)
        count += 1
        if count % 500 == 0:
            print("Segment {}/{} - frame {} - Start frame {} - End frame {}".format(current_segment+1, len(starts), 
                    count, starts[current_segment], ends[current_segment]))
        #if count > starts[0] + 1000: break

    if write: out.release()
    video.release()

    return True

def project_att(Rm, view_list, ac_label, curr_cam):
    angles = []
    Rm = rotation_matrix_from_vectors(np.array([0,0,1.0]), Rm)
    for v, ac in zip(view_list, ac_label):
        if ac == 0: 
            angles.append(np.array([0.0,0.0,0.0]))
            continue
        cam = curr_cam[v]
        A  = Rm @ cam["R"].T
        Cx = Rotation.from_rotvec( cam["R"].T[0] * np.radians(180)).as_matrix()
        Cy = Rotation.from_rotvec( cam["R"].T[1] * np.radians(180)).as_matrix()
        A  = A @ Cx @ Cy
        A = Rotation.from_matrix(A).as_euler("xyz",degrees = True)
        angles.append(A)
            
    return np.array(angles)

def rotation_matrix_from_vectors(a, b):
    # Normalize the input vectors
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)

    c = np.dot(a, b)

    if abs(c + 1.0) < 1e-6:
        # In this case, the vectors are exactly opposite, so we need a 180-degree rotation.
        # A 180-degree rotation matrix around any axis is -1 times the identity matrix.
        return -np.eye(3)

    if abs(c - 1.0) < 1e-6:
        # In this case, the vectors are already aligned, so no rotation is needed.
        return np.eye(3)

    v = np.cross(a, b)
    s = np.linalg.norm(v)

    # Skew-symmetric cross product matrix
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

    # Rodrigues' rotation formula
    rotation_matrix = np.eye(3) + kmat + np.dot(kmat, kmat) * ((1 - c) / (s ** 2))

    return rotation_matrix

if __name__ == '__main__':
    video_path = ""
    bbox_path = ""
    main_headpose(video_path, bbox_path, write = True)
