import os
import re

import cv2
import numpy as np
from math import cos, sin
from scipy.spatial.transform import Rotation  
from collections import OrderedDict

import matplotlib.pyplot as plt

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
            onset, offset = 29.97*onset/1000.0, 29.97*offset/1000.0
            record[data[1]][category].append((onset, offset))
            
    return record
    
def plot_pose_cube(img, yaw, pitch, roll, tdx=None, tdy=None, size=150.):
    # Input is a cv2 image
    # pose_params: (pitch, yaw, roll, tdx, tdy)
    # Where (tdx, tdy) is the translation of the face.
    # For pose we have [pitch yaw roll tdx tdy tdz scale_factor]

    p = pitch * np.pi / 180
    y = -(yaw * np.pi / 180)
    r = roll * np.pi / 180
    if tdx != None and tdy != None:
        face_x = tdx - 0.50 * size 
        face_y = tdy - 0.50 * size

    else:
        height, width = img.shape[:2]
        face_x = width / 2 - 0.5 * size
        face_y = height / 2 - 0.5 * size

    x1 = size * (cos(y) * cos(r)) + face_x
    y1 = size * (cos(p) * sin(r) + cos(r) * sin(p) * sin(y)) + face_y 
    x2 = size * (-cos(y) * sin(r)) + face_x
    y2 = size * (cos(p) * cos(r) - sin(p) * sin(y) * sin(r)) + face_y
    x3 = size * (sin(y)) + face_x
    y3 = size * (-cos(y) * sin(p)) + face_y


    # Draw base in red
    cv2.line(img, (int(face_x), int(face_y)), (int(x1),int(y1)),(0,0,255),3)
    cv2.line(img, (int(face_x), int(face_y)), (int(x2),int(y2)),(0,0,255),3)
    cv2.line(img, (int(x2), int(y2)), (int(x2+x1-face_x),int(y2+y1-face_y)),(0,0,255),3)
    cv2.line(img, (int(x1), int(y1)), (int(x1+x2-face_x),int(y1+y2-face_y)),(0,0,255),3)
    # Draw pillars in blue
    cv2.line(img, (int(face_x), int(face_y)), (int(x3),int(y3)),(255,0,0),2)
    cv2.line(img, (int(x1), int(y1)), (int(x1+x3-face_x),int(y1+y3-face_y)),(255,0,0),2)
    cv2.line(img, (int(x2), int(y2)), (int(x2+x3-face_x),int(y2+y3-face_y)),(255,0,0),2)
    cv2.line(img, (int(x2+x1-face_x),int(y2+y1-face_y)), (int(x3+x1+x2-2*face_x),int(y3+y2+y1-2*face_y)),(255,0,0),2)
    # Draw top in green
    cv2.line(img, (int(x3+x1-face_x),int(y3+y1-face_y)), (int(x3+x1+x2-2*face_x),int(y3+y2+y1-2*face_y)),(0,255,0),2)
    cv2.line(img, (int(x2+x3-face_x),int(y2+y3-face_y)), (int(x3+x1+x2-2*face_x),int(y3+y2+y1-2*face_y)),(0,255,0),2)
    cv2.line(img, (int(x3), int(y3)), (int(x3+x1-face_x),int(y3+y1-face_y)),(0,255,0),2)
    cv2.line(img, (int(x3), int(y3)), (int(x3+x2-face_x),int(y3+y2-face_y)),(0,255,0),2)

    return img

def draw_axis(img, yaw, pitch, roll, tdx=None, tdy=None, size = 100):
    center_x = img.shape[1] // 2
    center_y = img.shape[0] // 2
    if tdx != None and tdy != None:
        center_x = tdx
        center_y = tdy

    start_point = (center_x, center_y)

    M = Rotation.from_euler("xyz", np.array([yaw, pitch, roll]), degrees = True).as_matrix()
    # Define the 3D face direction vector (for example, in this case, it points right in the image plane)
    x = M[0] / np.linalg.norm(M[0])
    y = M[1] / np.linalg.norm(M[1])
    z = M[2] / np.linalg.norm(M[2])

    # Calculate the ending point based on the vector length (you may need to adjust the scaling)
    end_point_x = (start_point[0] + int(x[0] * size),
                start_point[1] + int(x[1] * size))
    end_point_y = (start_point[0] + int(y[0] * size),
                start_point[1] + int(y[1] * size))
    end_point_z = (start_point[0] + int(z[0] * size),
                start_point[1] + int(z[1] * size))

    # Draw the vector onto the background image
    thickness = 4  # Line thickness
    cv2.line(img, start_point, end_point_x, (0, 0, 255), thickness)
    cv2.line(img, start_point, end_point_y, (0, 255, 0), thickness)
    cv2.line(img, start_point, end_point_z, (255, 0, 0), thickness)
    return img

def get_gt(gtdir):
    video_list = {}
    for filename in os.listdir(gtdir):
        #if not "_HW_" in filename: continue
        sess_name = re.findall(r'\d\d_\d\d', filename)[0]
        if sess_name not in video_list: video_list[sess_name] = []
        video_list[sess_name].append(os.path.join(gtdir, filename))
    return video_list

def extract_corrected(filename):
    if not os.path.exists(filename): return
    attention = {}
    corrected_frames = {}
    with open(filename, "r") as f:
        data = f.readlines()

    for i, d in enumerate(data):
        d_split = d.replace("\n", "").split(",")
        if len(d_split)== 10:
            frame, b0, b1, b2, A0, A1, A2, att0, att1, att2 = d_split
            flag = 0
        elif len(d_split)== 11:
            frame, b0, b1, b2, A0, A1, A2, att0, att1, att2, flag = d_split
        elif len(d_split)== 18:
            frame, flag, flag_h, b0, b1, b2, A0, A1, A2, att0, att1, att2, xhl, yhl, zhl, xhr, yhr, zhr = d_split
        elif len(d_split) < 10: continue
        else:
            print("Error in attention file")
            exit()
        flag = int(flag)
        pos = np.array([float(att0), float(att1), float(att2)])
        A = [float(a) for a in [A0, A1, A2]]
        #vec = np.array([float(A0), float(A1), float(A2)])
        b = np.array([float(b0), float(b1), float(b2)])

        att_line = np.array([b, pos])
        size = np.linalg.norm(pos - b)
        if flag > 0: corrected_frames[int(frame)]= flag
        if size < 1e-6: 
            attention[int(frame)] = np.copy(attention[int(frame) - 1]).item()
            continue
        vec = (pos - b)/ ( size + 1e-6)
        att_vec = np.array([b, b + 1*vec]) 
        attention[int(frame)] = {"u":att_vec, "line":att_line, "head":b, "att":pos,
                                "size":size, "corrected_flag":flag, "A":A}  
    print("Attention Loaded with", len([x for x, y in corrected_frames.items() if y == 1]), "already corrected frames")
    print("Attention Loaded with", len([x for x, y in corrected_frames.items() if y == 2]), "frames selected for correction")
        
    return attention, corrected_frames

def plot_frame(img, p2d):
    for c, info in p2d.items():       
        pos = info["p"]   
        is_front = info["is_front"]  
        box = info["box"]  
        angles = info["A"] 
        scale = info["scale"]
        color = (0,255,0) if is_front else (0,0,255)
        img = cv2.circle(img, pos, radius=3, color= (255,0,0), thickness=10)
        img = cv2.circle(img, info["att"] , radius=3, color= (255,0,0), thickness=10)
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color= color)
        #img = plot_pose_cube(img, angles[0], angles[1], angles[2], tdx=pos[0], tdy=pos[1], size = int(200/scale))
        img = draw_axis(img, angles[0], angles[1], angles[2], tdx=pos[0], tdy=pos[1], size = int(200/scale))
    return img

def project_2d(poses, cams, h, w):
    hh, ww = h//4, w//2

    p3d = poses["head"]
    p2d_list = {}
    offset = 50
    for c, cam in cams.items():
        has_head = True
        u = cam["u"]
        t = -cam["R"] @ cam["T"]
        p2d, _ = cv2.projectPoints(np.array([p3d, poses["att"]]).T, cam["r"], t, cam["mtx"], cam["dist"])
        p2d = p2d.reshape(-1,2)
        
        if not (offset < p2d[0,0] < ww -offset and offset < p2d[0,1] < hh - offset): has_head = False

        if c == 1: p2d[:,0] += ww
        elif c == 2: p2d[:,1] += hh
        elif c == 3:  p2d += np.array([ww, hh])
        elif c == 4:  p2d[:,1] += 2*hh
        elif c == 5:  p2d += np.array([ww, 2*hh])
        elif c == 6:  p2d[:,1] += 3*hh
        elif c == 7:  p2d += np.array([ww, 3*hh])

        att = p2d[1].astype("int")
        p2d = p2d[0]
        
        if has_head: 
            s = np.linalg.norm(t - np.array(p3d))
            x1, x2 = max(0, p2d[0] - 200/s), min(w, p2d[0] + 200/s)
            y1, y2 = max(0, p2d[1] - 200/s), min(h, p2d[1] + 200/s)
            box = np.array([x1,y1, x2,y2]).astype("int")
            att_dir = poses["u"][1] - poses["u"][0]
            att_dir = att_dir/np.linalg.norm(att_dir)
            is_front = np.dot(u, att_dir) < 0

            M = rotation_matrix_from_vectors(np.array([0,0,1.0]), att_dir)
            A  = M @ cam["R"].T
            Cx = Rotation.from_rotvec( cam["R"].T[0] * np.radians(180)).as_matrix()
            Cy = Rotation.from_rotvec( cam["R"].T[1] * np.radians(180)).as_matrix()
            A  = A @ Cx @ Cy
            A = Rotation.from_matrix(A).as_euler("xyz",degrees = True)
  
                    
            p2d_list[c] = {"p":p2d.astype("int"), "scale":s, "box":box, "is_front":is_front, "A":A, "att_dir":att_dir}


    return p2d_list

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

def is_point_inside_box(point, box):
    x, y = point
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2

def calculate_iou(bbox1, bbox2):
    """
    Calculate the Intersection over Union (IoU) between two bounding boxes.

    Parameters:
        bbox1 (list or tuple): Coordinates of the first bounding box in the format (x1, y1, x2, y2).
        bbox2 (list or tuple): Coordinates of the second bounding box in the format (x1, y1, x2, y2).

    Returns:
        float: Intersection over Union (IoU) score.
    """
    x1_i = max(bbox1[0], bbox2[0])
    y1_i = max(bbox1[1], bbox2[1])
    x2_i = min(bbox1[2], bbox2[2])
    y2_i = min(bbox1[3], bbox2[3])

    intersection_area = max(0, x2_i - x1_i + 1) * max(0, y2_i - y1_i + 1)

    area_bbox1 = (bbox1[2] - bbox1[0] + 1) * (bbox1[3] - bbox1[1] + 1)
    area_bbox2 = (bbox2[2] - bbox2[0] + 1) * (bbox2[3] - bbox2[1] + 1)

    union_area = area_bbox1 + area_bbox2 - intersection_area

    iou = intersection_area / union_area if union_area > 0 else 0.0
    return iou

def check_bb_list(p2d_info, view, bbox):
    views = bbox["views"]
    if not view in views: return None
    
    proposal, iou_list = [], []
    for bb, v in zip( bbox["bbox"],  views):
        if v != view: continue
        if is_point_inside_box(p2d_info["p"], bb[:4]):
            iou = calculate_iou(bb[:4], p2d_info["box"])
            if iou < 0.2: continue
            proposal.append(bb[:4])  
            iou_list.append(iou)    
    if len(proposal) > 0:
         best = np.argmax(iou)
         return proposal[best]
    return None

def get_bbox(p2d_info, view, frame_id, data):
    best_box = None
    for i in range(3):
        if frame_id + i in data:
            best_box = check_bb_list(p2d_info, view, data[frame_id+i])
        if best_box is not None: break
        
        if frame_id - i in data:
            best_box = check_bb_list(p2d_info, view, data[frame_id-i])
        if best_box is not None: break
    return best_box

def generate_helper(frame, mydict):
    frame_id=mydict['frame_id']
    data=mydict['data']
    p2d=mydict['p2d']
    name1=mydict['sess']
    gt_dir=mydict['gt_dir']
    p3d=mydict['p3d']

    for c, info in p2d.items():
        if mydict["is_mat_view"] and c not in [0,1,2,3,6]: continue
        box = get_bbox(info, c, frame_id, data)
        if box is None: 
            box = info["box"]
            
        x1, y1, x2, y2 = box
        is_front = info["is_front"]
        yaw, pitch, roll = info["att_dir"]
        name = "{}_{}_{}.png".format(name1,frame_id,c)
        patch = frame[y1:y2, x1:x2]
        
        img_path = os.path.join(output_dir, name)
        #if os.path.exists(img_path): continue
        cv2.imwrite(img_path, patch)
        #draw_axis(patch, yaw, pitch, roll, tdx=None, tdy=None, size = 50)
        with open(gt_dir, "a") as f:
            f.write("{},{},{},{},{},{},{},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f}\n".format(name, mydict['type'], 
                    int(is_front),int(x1), int(y1), int(x2), int(y2), yaw, pitch, roll, p3d[0], p3d[1], p3d[2]))


def generate_dataset(mydict, gtdir, timestamps, output_dir = ""):
    videodir = mydict['videodir']
    bodydir = mydict['bodydir']
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    video_list = get_gt(gtdir)
    timestamps = get_timestamp(timestamps)
    # Camera Files
    cams_room = np.load("camera_room.npy", allow_pickle=True).item()
    cams_mat = np.load("camera_mat.npy", allow_pickle=True).item()

    gt_dir= os.path.join(output_dir, "../gt_body_quad.txt")
    mydict['gt_dir']=gt_dir
    

    with open(gt_dir, "w") as f:
        f.write("")

    count = 0
    for sess, list_gt in video_list.items():
        #if sess not in ["07_04", "19_03", "15_04", "20_06", "24_02"]: continue
        videopath = os.path.join(videodir, "DevEv_S" + sess + "_Sync.mp4")
        if not os.path.exists(videopath):             
            print("File not found", videopath)
            exit()
        time_info = timestamps[sess]
        vidcap = cv2.VideoCapture(videopath)
        res, frame = vidcap.read()
        h, w, _ = frame.shape
    
        for k, gtfile in enumerate(list_gt):
            path1=bodydir+'data_2d_DevEv_S{}_Sync.npz'.format(sess)
            if not os.path.exists(path1):
                print("File not found", path1)
                exit()
            data=np.load(path1, allow_pickle=True)['data'].item()
            mydict['data']=data
            mydict['sess']=sess

            attention, corrected_frames = extract_corrected(gtfile)
            frame_list = sorted(list(corrected_frames.keys()))
            last_corrected = frame_list[0]
            for f in frame_list[1:]:
                if abs(last_corrected - f) <= 30:
                    for i in range(last_corrected, f):
                        corrected_frames[i] = 1
                last_corrected = f
            

            print("{} current {}/{}:".format(sess, count, len(video_list)))
            for (frame_id, is_corrected) in corrected_frames.items():
                if is_corrected != 1: continue
                vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_id-1)                
                cams, is_mat = get_cams(frame_id, time_info, cams_room, cams_mat)
                mydict['is_mat_view'] = is_mat
                mydict['frame_id_base']=frame_id
                     
                res, frame = vidcap.read()
                if not res: 
                    print("error reading frame", frame_id)
                    exit()
                #print(frame_id, curr_frame)
                if frame_id not in attention: continue
                info = attention[frame_id]
                p2d = project_2d(info, cams, h, w)    
                mydict['u'] = info["u"]                   
                mydict['p2d'] = p2d
                mydict['frame_id'] = frame_id
                mydict['p3d'] = info["head"]
                mydict['type'] = 0
                generate_helper(frame, mydict)  
                # return 
            count += 1
        vidcap.release()
    return

def get_cams(frame_id, time_info, cams_room, cams_mat):
    if 'c' in time_info:
        for s, e in time_info['c']:
            if s <= frame_id <=e : return cams_mat, 1
    return cams_room, 0
    

def extract_unbalance_dataset(mydict, gtdir, timestamps, output_dir = ""):
    videodir = mydict['videodir']
    with open("../6DRepNet/data_imbalance.txt", "r") as f:
        video_list = f.readlines()


    for info in video_list:
        quadrant, total, sess = info.split(",")
        sess, subj, frame = sess.split("_")
        sess = sess + "_" + subj
        name = "{}_{}_{}.png".format(quadrant, sess, frame)
        #if sess not in ["07_04", "19_03", "15_04", "20_06", "24_02"]: continue
        videopath = os.path.join(videodir, "DevEv_S" + sess + "_Sync.mp4")
        if not os.path.exists(videopath):             
            print("File not found", videopath)
            exit()

        vidcap = cv2.VideoCapture(videopath)
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, int(frame) - 1)      
        res, img = vidcap.read()    
        cv2.imwrite("attention_results/" + name, img) 
        vidcap.release()
    return


if __name__ == "__main__":
    gtdir = "corrected_quadrant/"
    output_dir = "/nfs/hpc/cn-gpu5/DevEv/headpose_dataset/bodyhead_dataset_quad/"
    videodir = "/nfs/hpc/cn-gpu5/DevEv/dataset/"
    bodydir = "/nfs/hpc/cn-gpu5/DevEv/viz_bodypose/"
    timestamps = "DevEvData_2023-06-20.csv"
    mydict={'bodydir':bodydir, 'videodir':videodir}
    
    #extract_unbalance_dataset(mydict, gtdir, timestamps, output_dir = "")
    generate_dataset(mydict, gtdir, timestamps, output_dir = output_dir)

