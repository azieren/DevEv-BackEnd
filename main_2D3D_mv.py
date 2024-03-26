import os
import re
import numpy as np
import cv2
from scipy import interpolate
import argparse

import trimesh
from scipy.spatial.transform import Rotation as R
from scipy.spatial import distance_matrix
from scipy.ndimage import gaussian_filter as filter1d

from databrary import get_videos, get_processed_head

SUBLIST = ["16_04", "24_02", "26_06", "27_03", "27_06", "28_01", "29_02", "29_03", "29_07"]
SUBLIST = ["20_02","20_03","26_06", "27_03", "29_01", "29_03", "29_09", "33_01", "34_02", "34_03", "35_01"]
SUBLIST = ["05_04", "07_04","08_03", "13_03", "13_04", "14_03", "14_06", "15_03", "15_04", "15_09", 
           "15_10", "15_11", "16_04", "16_06", "18_01","18_05", "19_01", "19_02", "19_03", "19_04", 
           "19_06", "20_02", "20_03", "20_05", "20_06", "20_09", "21_03", "21_06", "21_08", 
           "24_02", "25_01", "26_02", "26_05", "26_06", "27_01", "27_02", "27_03", "27_06", 
           "28_01", "29_01", "29_02", "29_03", "29_07", "33_01", "34_02", "34_03", "34_04", "35_01"]

#SUBLIST = ["15_10", "19_02", "15_04", "20_06", "07_04", "15_04"]
#SUBLIST = ["07_04", "08_03", "13_03"] 

PARENT_BOX_B_1 = [685, 418, 960, 540]
PARENT_BOX_B_2 = [314, 0, 500, 127]

PARENT_BOX_M_1 = [175, 49, 251, 127]
PARENT_BOX_M_2 = [619, 115, 683, 184]

def extract_video_info(info, timestamps):
    video = os.path.join("/nfs/hpc/cn-gpu5/DevEv/dataset/", "{}.{}".format(info["download"]["name"], info["download"]["format"]))
    video = cv2.VideoCapture(video)
    # New cv2
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))   # float
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)) # float
    fps = video.get(cv2.CAP_PROP_FPS)
    video.release()
    
    starts, ends, cams = [], [], []
    for n, info in timestamps.items():
        for (s, e) in info:
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
    return width, height, starts, ends, cams

def read_headposes(posefile, h, w):
    h, w = h//4, w//2
    poses = {}

    with open(posefile, "r") as f:
        lines = f.readlines()
    count = 0
    for l in lines:
        frame, flag, x1, y1, x2, y2, yaw, pitch, roll = l.split(',')
        frame, flag = int(eval(frame)), int(eval(flag))
        bb = [int(eval(x1)), int(eval(y1)),int(eval(x2)), int(eval(y2))]
        #head_box = [int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])]
        x_min, y_min, x_max, y_max = bb
        yaw, pitch, roll = float(yaw), float(pitch), float(roll)

        x, y = (x_min + x_max) / 2, (y_min + y_max) / 2
        area = (x_max - x_min) * (y_max - y_min)
        
        

        if x <= w and y <= h:
            c_ = 0 
        elif x > w and y <= h:
            c_ = 1
            x -= w
        elif x <= w and 2*h >= y > h:
            c_ = 2
            y -= h
        elif x > w and 2*h >= y > h:
            c_ = 3      
            x -= w
            y -= h
        elif x <= w and 3*h >= y > 2*h:
            c_ = 4      
            y -= 2*h
        elif x > w and 3*h >= y > 2*h:
            c_ = 5  
            x -= w    
            y -= 2*h
        elif x <= w and y > 3*h:
            c_ = 6  
            y -= 3*h
        else:
            c_ = 7
            x -= w   
            y -= 3*h

        #if c_ in [1] and (PARENT_BOX_M_1[0]  <= x <= PARENT_BOX_M_1[2] and PARENT_BOX_M_1[1] <= y <= PARENT_BOX_M_1[3]) : continue
        #if c_ in [2] and (PARENT_BOX_M_2[0]  <= x <= PARENT_BOX_M_2[2] and PARENT_BOX_M_2[1] <= y <= PARENT_BOX_M_2[3]) : continue

        #if c_ in [5] and (PARENT_BOX_B_1[0]  <= x <= PARENT_BOX_B_1[2] and PARENT_BOX_B_1[1] <= y <= PARENT_BOX_B_1[3]) : continue
        #if c_ in [6] and (PARENT_BOX_B_2[0]  <= x <= PARENT_BOX_B_2[2] and PARENT_BOX_B_2[1] <= y <= PARENT_BOX_B_2[3]) : continue
        if c_ in [5] and (y <= 120) : continue
        
        ## Remove edges
        margin = 30
        if x < margin or x > w - margin or y < margin or y > h - margin: 
            continue

        if frame not in poses:
            poses[frame] = {}   
        if flag == 0: continue
        
        if c_ in poses[frame]:
            old_area = poses[frame][c_]["area"]
            if old_area > area: continue
        record = {"pos":[x, y], "angle": [yaw, pitch, roll], "c":c_, "flag":flag, "area":area}
        poses[frame][c_] = record
        
        if count % 5000 == 0:
            print(count)
        #if count > 10000: break
        count += 1
    
    return poses

def read_bodyposes(poses, bodyfile, height, width):
    bodyfile = bodyfile.replace("output_", "data_2d_").replace(".txt", ".npz")
    if not os.path.exists(bodyfile):
        print(bodyfile, "not found")
        exit()
    bodyinfo = np.load(bodyfile, allow_pickle=True)
    bodyinfo = bodyinfo["data"].item()

    to_delete = []
    for f, views in poses.items():
        if not f in bodyinfo: 
            print("frame not in body")
            exit()
        body = bodyinfo[f]
        body_views = np.array(body["views"], dtype=int)
        body_kpt = np.array(body["kpt"], dtype=int).astype(float)
        for v, info in views.items():
            if v not in body_views:
                #info["left_hand"] = poses[old_f][v]["left_hand"]
                #info["right_hand"] = poses[old_f][v]["right_hand"]
                to_delete.append((f,v))
                continue

            #0: 'nose',
            #1: 'left_eye',
            #2: 'right_eye',
            body_kpt_v = body_kpt[body_views == v,:,:2]
            p2d_head = info["pos"]
            nose = (body_kpt_v[:,1] + body_kpt_v[:,2])/2.0
            best = np.abs(p2d_head - nose).sum(1)
            best = np.argmin(best)
            kpt = body_kpt_v[best]

            #    9: 'left_wrist', 10: 'right_wrist'  /   15: 'left_ankle', 16: 'right_ankle'
            info["left_hand"] = kpt[9]
            info["right_hand"] = kpt[10]
            
            info["left_ankle"] = kpt[15]
            info["right_ankle"] = kpt[16]  
            info["pos"] = (kpt[0] + kpt[1])/2.0


    for (f,v) in to_delete:
        del poses[f][v]          
    return poses

def interpolate2d(camera, frame_list, poses, threshold_frame = 40, threshold_pos = 50):
    if len(frame_list) <= 1: return poses
    diff = frame_list[-1] - frame_list[-2]   
    if diff > threshold_frame or diff == 1: return poses
    start, end = frame_list[-2], frame_list[-1]
    area, flag = poses[start][camera]["area"], poses[start][camera]["flag"]
    p0, p1 = poses[start][camera]["pos"], poses[end][camera]["pos"]
    diff_pos = ((np.array(p0) - np.array(p1))**2).sum()
    if diff_pos >= threshold_pos: return poses
    
    a0, a1 = poses[start][camera]["angle"], poses[end][camera]["angle"]
    
    for i, f in enumerate(range(start+1, end)):
        t = (i+1)/diff
        p = t*np.array(p0) + (1-t)*np.array(p1)
        a = t*np.array(a0) + (1-t)*np.array(a1)
        p = p.astype(int)
        a = a.astype(int)
        r = {"pos":[p[0], p[1]], "angle": [a[0], a[1], a[2]], "c":camera, "flag":flag, "area":area}
        if not f in poses: poses[f] = {}
        poses[f][camera] = r
    return poses

def to_3D(p0, p1, c0, c1):
    p0 = cv2.undistortPoints(p0.reshape((1,1,2)), c0["mtx"], c0["dist"], None, c0["mtx"])
    p1 = cv2.undistortPoints(p1.reshape((1,1,2)), c1["mtx"], c1["dist"], None, c1["mtx"])
    p4d = cv2.triangulatePoints(c0["K"], c1["K"], p0, p1)
    p3d = (p4d[:3, :]/p4d[3, :]).T
    return p3d.reshape(3)

def get_headpose(angle):
    r = R.from_euler('xyz', angle, degrees=False).as_matrix()
    v3 = [0, 0, 1]
    return np.dot(r,v3)

def to_3D_pose(points, cameras, height, width):
    C, C_, P = [], [], []
    handL, handR = [], []
    h, w = height//4, width//2
    track = list(points.keys())

    cost = []
    for t in track:
        if t == 10: continue
        p = points[t]

        angle = np.array(p["angle"])
        #angle = get_headpose(angles)
        
        x, y = p['pos']
        #print(t, x, y, w, h, height, width)
        cost_score = np.sqrt((x/w-0.5)**2 + (y/h-0.5)**2)
        cost.append(cost_score)

        c = cameras[p["c"]]
        P.append(p['pos'])
        C.append(c)
        C_.append(p["c"])
        handL.append(p['left_hand'])
        handR.append(p['right_hand'])

    if len(P) == 1: return C_, None, angle, None, None
    elif len(P) == 0: return [], None, None, None, None
    

    ind = np.argsort(cost)
    C_ = [C_[i] for i in ind ]
    C = [C[i] for i in ind ]
    P = np.array([P[i] for i in ind ], dtype=float)
    cost = [cost[i] for i in ind]   
    handL = [handL[i] for i in ind]
    handR = [handR[i] for i in ind]

    if len(P) == 2: 
        p3d = to_3D(P[0], P[1], C[0], C[1])
        pair = (0,1)
    else:
        p3d, pair = outlier_remove_to_3D(P, C)
        #p3d = to_3D(P[ind[0]], P[ind[1]], C[ind[0]], C[ind[1]])
        #pair = (ind[0], ind[1])
    
    p0, p1 = pair
    handL = to_3D(handL[p0], handL[p1], C[p0], C[p1])
    handR = to_3D(handR[p0], handR[p1], C[p0], C[p1])

    if np.linalg.norm(p3d - handL) > 0.5:
        handL = p3d + 0.5*(handL - p3d)/ np.linalg.norm(handL - p3d)
    if np.linalg.norm(p3d - handR) > 0.5:
        handR = p3d + 0.5*(handR - p3d)/ np.linalg.norm(handR - p3d)
    #print(p3d.shape, p3d)
    return C_, p3d, angle, handL, handR

def outlier_remove_to_3D(P, C):
    num_points = min(5, len(P))
    p3d_list = []
    pair_list = []
    for i in range(num_points):
        for j in range(i + 1, num_points):
            # Perform triangulation for each combination
            triangulated_points = to_3D(P[i], P[j], C[i], C[j])
            p3d_list.append(triangulated_points)
            pair_list.append([i,j])
    pair_list = np.array(pair_list)
    p3d_list = np.array(p3d_list)
    d = distance_matrix(p3d_list, p3d_list)
    bestscore = np.argmin(d.sum(-1)/(len(p3d_list) - 1.0))
    return p3d_list[bestscore], pair_list[bestscore]

def smooth_3D(poses_3D, directions, filter_t = 7.0, filter_i = np.inf):
    frame_list = sorted(list(poses_3D.keys()))

    frames = list(range(min(frame_list), max(frame_list) +1 ))

    data_o, data_p, data_hl, data_hr = [], [], [], []

    # Interpolate
    for i in range(len(frame_list)):
        f = frame_list[i]
        record =  poses_3D[f]

        data_p.append(record["pos_3d"])
        data_hl.append(record["handL"])
        data_hr.append(record["handR"])
        if not np.isnan(record["orientation"].sum()):
            data_o.append(record["orientation"])
        else:
            if len(data_o) == 0: 
                data_o.append(poses_3D[frame_list[i+1]]["orientation"])
            data_o.append(data_o[-1])

    data_o, data_p = np.array(data_o), np.array(data_p)
    data_hl, data_hr = np.array(data_hl), np.array(data_hr)
    
    interpolate_p = interpolate.interp1d(frame_list, data_p.T, kind = 'quadratic')
    data_p = interpolate_p(frames).T
    interpolate_p = interpolate.interp1d(frame_list, data_hl.T, kind = 'quadratic')
    data_hl = interpolate_p(frames).T
    interpolate_p = interpolate.interp1d(frame_list, data_hr.T, kind = 'quadratic')
    data_hr = interpolate_p(frames).T
    interpolate_o = interpolate.interp1d(frame_list, data_o.T, kind = 'quadratic')
    data_o = interpolate_o(frames).T 
       
    # Smooth
    for i in range(3):
        data_p[:,i] = filter1d(data_p[:,i], filter_t)
        data_hl[:,i] = filter1d(data_hl[:,i], 3)
        data_hr[:,i] = filter1d(data_hr[:,i], 3)
        data_o[:,i] = filter1d(data_o[:,i], filter_t)

    print(len(data_o), len(frame_list), frames[0], frames[-1])
    for i, f in enumerate(frames):
        if f not in poses_3D: poses_3D[f] = {}      
        poses_3D[f]["pos_3d"] = data_p[i]
        poses_3D[f]["orientation"] = data_o[i] / np.linalg.norm(data_o[i], axis = 0)
        poses_3D[f]["pos_2d"] = []
        poses_3D[f]["handL"] = data_hl[i]
        poses_3D[f]["handR"] = data_hr[i]
        
    return poses_3D

def process_view_count(video_info):
    hist = {i:0 for i in range(8)}
    for name, info in video_info.items():
        # Select only first half of available files
        if not "download" in info: 
            print(name, " could not download")
            continue
        if not "head" in info: continue
        if not info["head"]: continue
        
        print(name)
        # Download video
        timestamps = {key: info[key] for key in ['c', 'r', 'p'] if key in info}
        new_hist = count_views(info, timestamps)
        for i in range(8):
            hist[i] += new_hist[i]

    total = sum([hist[i] for i in range(8)])
    for i in range(8):
        print("For ", i,  " elements in views: ", hist[i], hist[i]/total)

    return

def count_views(info, timestamps):
    width, height, starts, ends, cams = extract_video_info(info, timestamps)
    poses = read_headposes(info["head"], height, width)
    poses = read_bodyposes(poses, info["body"], height, width)

    current_segment = 0
    frame_list = list(poses.keys())
    sorted(frame_list)        

    hist = {i:0 for i in range(8)}
    for frame in frame_list:  
        #if frame < 22000: continue
        #print(frame) 
        if frame > ends[-1]: 
            print(frame)
            break
        
        if frame > ends[current_segment]:
            current_segment += 1  
           
        if not starts[current_segment] < frame: continue
        p = poses[frame] 
        
        nviews = min(7, len(p))
        hist[nviews] += 1
            
    return hist

def main_3dproj(info, timestamps,  output_dir = "output/"):
    cam_mat = np.load("camera_mat.npy", allow_pickle = True).item()
    cam_room = np.load("camera_room.npy", allow_pickle = True).item()

    width, height, starts, ends, cams = extract_video_info(info, timestamps)
    output_file = os.path.join(output_dir, "attC_" + info["download"]["name"] + ".txt")

    poses = read_headposes(info["head"], height, width)
    poses = read_bodyposes(poses, info["body"], height, width)
    poses_3D = {}

    current_segment = 0
    curr_cam = cam_room
    if cams[current_segment] == 'c':
        curr_cam = cam_mat
    frame_list = list(poses.keys())
    sorted(frame_list)        
    
    directions = [c["u"] for i, c in curr_cam.items()]  
    old_info = {"head":None, "angle":None, "handL":None, "handR":None}
    for frame in frame_list:  
        #if frame < 22000: continue
        #print(frame) 
        if frame > ends[-1]: 
            print(frame)
            break
        
        if frame > ends[current_segment]:
            current_segment += 1
            curr_cam = cam_room
            if cams[current_segment] == 'c':
                curr_cam = cam_mat      
           
        if not starts[current_segment] < frame: continue
        p = poses[frame] 
        C_, p3d, A, handL, handR = to_3D_pose(p, curr_cam, height, width)
        if len(C_) <= 1:
            p3d = p3d if p3d is not None else np.copy(old_info["head"])
            A = A if A is not None else np.copy(old_info["angle"])
            handL = old_info["handL"] if handL is not None else np.copy(old_info["handL"])
            handR = old_info["handR"] if handR is not None else np.copy(old_info["handR"])
        # Check high motion
        if old_info["head"] is not None and np.linalg.norm(old_info["head"] - p3d) > 0.3: 
            p3d = old_info["head"] + 0.3*(p3d-old_info["head"])
            #A = old_info["angle"] + 0.3*(p3d-old_info["angle"])
            handL = old_info["handL"] + 0.3*(p3d-old_info["handL"])
            handR = old_info["handR"] + 0.3*(p3d-old_info["handR"])
        elif old_info["head"] is None and p3d is None: continue
            
        if len(C_) <= 1 and len(poses_3D) == 0: continue

        # Check border of Room
        p3d[0] = max(p3d[0], -6)
        p3d[0] = min(p3d[0], 6)
        p3d[2] = max(p3d[2], 0.01)
        p3d[2] = min(p3d[2], 1.4)

        poses_3D[frame] = {'pos_3d':p3d, 'pos_2d':p, 'orientation':A, "handL":handL, "handR":handR}
        if frame % 1000 == 0:
            print(current_segment, cams[current_segment], starts[current_segment], ends[current_segment], frame)
        if np.isnan(p3d).any():
            print("Exit because of invalid value", frame, C_)
            exit()
        old_info = {"head":p3d, "angle":A, "handL":handL, "handR":handR}

    for s, e in zip(starts, ends):
        p_temp = {k:v for k, v in poses_3D.items() if s <= k <= e}
        if len(p_temp) == 0: continue
        p_temp = smooth_3D(p_temp, directions, filter_t = 7, filter_i = np.inf)
        poses_3D.update(p_temp)
    write_attention(poses_3D, output_file)
    return 

def write_attention(poses_3D, output_file):
    count = 0
    frames = np.sort(list(poses_3D.keys()))
    att = None
    mesh = trimesh.load_mesh('Room.ply')

    with open(output_file, "w") as f:

        for frame in frames:
            record = poses_3D[frame]
            b, A = record["pos_3d"], record["orientation"]
            handL, handR = record["handL"], record["handR"]
            #print("Find {} bbox in frame {}".format(len(record["pos_2d"]), frame))
            if np.linalg.norm(A) > 1e-6: 
                # load mesh
                # create some rays
                ray_origins = b.reshape(-1, 3)
                ray_directions = A.reshape(-1, 3)

                # Get the intersections
                intersection, index_ray, index_tri = mesh.ray.intersects_location(
                ray_origins=ray_origins, ray_directions=ray_directions)
   
                if len(intersection) > 0:
                    d = np.sqrt(np.sum((b - intersection) ** 2, axis=1))
                    ind = np.argsort(d)
                    att = intersection[ind[0]]
                else:
                    print("No intersection", frame)                 
            else:
                print("Norm wrong" ,frame, A)
                exit()

            if att is None: continue
            if np.nan in att:
                print("att has nan", frame, att, b)
                exit()

            f.write("{:d},{:d},{:d},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f}\n".format(
                frame, 0, 0, b[0], b[1], b[2], A[0], A[1], A[2], att[0], att[1], att[2], handL[0], handL[1], handL[2], handR[0], handR[1], handR[2]
                ))
            count += 1
            if count % 500 == 0:
                print("writing", count)

        return
    
def process_attention(video_info, path_processed, whitelist=None):
    if not os.path.exists(path_processed):
        os.makedirs(path_processed)

    for name, info in video_info.items():
        # Select only first half of available files
        #if not name in SUBLIST: continue
        if not "download" in info: 
            print(name, " could not download")
            continue
        if not "head" in info: continue
        if not info["head"]: continue
        if whitelist is not None:
            if not name in whitelist: continue
        else:
            # Check if video has been already processed
            processed = get_processed_head(path=path_processed)
            if any(name in p for p in processed): continue
        
        # Download video
        timestamps = {key: info[key] for key in ['c', 'r', 'p'] if key in info}
        if len(timestamps) == 0 : continue
        print(name, timestamps)
        print(info)
        #exit()
        main_3dproj(info, timestamps, output_dir = path_processed)
        print("Finished", name)
    return
     
def add_body_head(video_info, path_body, path_head):
    no_downloads = []
    for name, info in video_info.items():
        if not "download" in info: 
            print(name, " could not download")
            no_downloads.append(name)
            continue
        video = info["download"]["name"]
        bodyfile = os.path.join(path_body, "data_2d_{}.npz".format(video))
        headfile = os.path.join(path_head, "head-{}.txt".format(video))
        #print(name, os.path.exists(bodyfile), os.path.exists(headfile))
        #print(bodyfile, headfile)
        if not os.path.exists(bodyfile) or not os.path.exists(headfile): continue    
        info["head"] = headfile
        info["body"] = os.path.join(path_body, "output_{}.txt".format(video))
    return video_info

def check_timestamps(video_info, target_dir, path_processed):
    import cv2
    
    wrong_files = []
    for name, info in video_info.items():
        if not "download" in info: 
            print(name, " could not download")
            wrong_files.append((name, " could not download"))
            continue

        bodytxt = os.path.join(path_processed, "attC_" + info["download"]["name"] + ".txt")
        video_output = os.path.join(target_dir, "{}.{}".format(info["download"]["name"], info["download"]["format"]))
        vidcap = cv2.VideoCapture(video_output)
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        timestamps = {key: info[key] for key in ['c', 'r', 'p'] if key in info}
        if len(timestamps) == 0:
            wrong_files.append((name, "no timestamps"))
            continue
        if not os.path.exists(bodytxt): 
            wrong_files.append((name, "no att file"))
            continue
                
        with open(bodytxt, "r") as f:
            body = f.readlines()        
        frames = set([int(x.split(",")[0]) for x in body if len(x) > 0])
        
        for n, stamps in timestamps.items():
            for (s,e) in stamps:
                start, end = int(fps*s/1000), int(fps*e/1000)
                if end - start < 10: continue
                sublist_frames = sorted([f for f in frames if start<=f<=end])
                if len(sublist_frames) == 0: diff = 0.0
                else:
                    L = max(sublist_frames) - min(sublist_frames)
                    diff = 100*L/ (end-start+1)
                if diff < 95.0: 
                    wrong_files.append((name, diff))
                    print("Last wrong")
                    print(info["download"]["name"], n, start, end, diff, len(sublist_frames))
                    continue
                
    wrong_files = list(set(wrong_files))
    print(wrong_files)
    return wrong_files


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default="/nfs/hpc/cn-gpu5/DevEv/viz_attention/", help="Directory path where 3D pose files will be written")
    parser.add_argument('--body_dir', type=str, default="/nfs/hpc/cn-gpu5/DevEv/viz_bodypose/", help="Directory path where body pose files are")
    parser.add_argument('--head_dir', type=str, default="/nfs/hpc/cn-gpu5/DevEv/viz_headpose/", help="Directory path where head pose files are")
    parser.add_argument('--timestamps', type=str, default="DevEvData_2024-02-02.csv", help="Path to timestamp file")
    parser.add_argument('--uname', type=str, default="azieren@oregonstate.edu", help="Databrary username")
    parser.add_argument('--psswd', type=str, default="changetheworld38", help="Databrary password")
    parser.add_argument('--check_time', action="store_true", help="Used only for checking the current amount of frames processed by existing files")
    parser.add_argument('--video_dir', type=str, default="/nfs/hpc/cn-gpu5/DevEv/dataset/", help="Directory path containing original videos. Only used with --check_time")
    parser.add_argument('--session', default = "", type=str, help="If used, only this session will be processed. Format: session and subject number ##_##")
    args = parser.parse_args()
    
    
    if args.uname == "":
        print("Enter a Databrary Username")
        exit()
    if args.uname == "":
        print("Enter a Databrary Password")
        exit()
    sess_name = re.findall(r'\d\d_\d\d', args.session)
    if len(sess_name) == 0:
        args.session = None
    else:
        args.session = sess_name[0]
    return args



if __name__ == "__main__":
    args = parse_args()
    video_info, session = get_videos(args.uname, args.psswd, args.timestamps)
    video_info = add_body_head(video_info, args.body_dir, args.head_dir)

    if args.check_time:
        check_timestamps(video_info, args.video_dir, args.output_dir)
        exit()    
    process_attention(video_info, args.output_dir, whitelist = args.session)
    
    #process_view_count(video_info)