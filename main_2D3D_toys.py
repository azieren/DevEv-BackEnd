import os
import re
import numpy as np
import cv2
from scipy import interpolate
import argparse
import copy
from scipy.spatial import distance_matrix
from scipy.ndimage import gaussian_filter as filter1d

from databrary import get_videos

CLASSES = [
    'pink_ballon', 'tree', 'pig', 'red_tower', 'farm', 'xylophone', 'cart', 'stroller', 'tower', 'bucket',
]

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

def read_toys(path):
    data = np.load(path, allow_pickle=True).item()
    return data

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

def to_3D(p0, p1, c0, c1):
    p0 = cv2.undistortPoints(p0.reshape((1,1,2)), c0["mtx"], c0["dist"], None, c0["mtx"])
    p1 = cv2.undistortPoints(p1.reshape((1,1,2)), c1["mtx"], c1["dist"], None, c1["mtx"])
    p4d = cv2.triangulatePoints(c0["K"], c1["K"], p0, p1)
    p3d = (p4d[:3, :]/p4d[3, :]).T
    return p3d.reshape(3)

def to_3D_pose(frame_info, cameras, height, width):
    
    h, w = height//4, width//2
    
    data_toys = {}
    for toy in CLASSES:
        data_toys[toy] = {}
        C, P, cost, track_list = [], [], [], {}
        for view in range(8):
            if not view in frame_info: continue
            if len(frame_info[view]["scores"]) == 0: continue
            index = [ i for i, label in enumerate(frame_info[view]["label_names"] ) if toy in label]
            if len(index) == 0: continue
            scores = np.array(frame_info[view]["scores"])
            best = np.argmax(scores[index])
            best = index[best]
            x1, y1, x2, y2 = frame_info[view]["bboxes"][best]
            score = scores[best]
            x, y = (x1+x2)/2.0, (y1+y2)/2.0
            cost_score = np.sqrt((x/w-0.5)**2 + (y/h-0.5)**2)
            P.append(np.array([x, y]))
            C.append(view)
            cost.append(cost_score)
            track_list[view] = frame_info[view]["track_ids"][best]
        if len(P) <= 1: continue  
         
        ind = np.argsort(cost)
        P = [P[i] for i in ind]
        C = [cameras[C[i]] for i in ind]
        if len(P) >= 3:
            p3d, _ = outlier_remove_to_3D(P, C)
        else: p3d = to_3D(P[0], P[1], C[0], C[1])

        p3d[0] = max(p3d[0], -6)
        p3d[0] = min(p3d[0], 6)
        p3d[2] = max(p3d[2], 0.01)
        p3d[2] = min(p3d[2], 1.4)
        
        data_toys[toy] = {"p3d":p3d, "track":track_list}
 
    return data_toys

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

def smooth_3D(poses_3D, filter_t = 7.0, theta_v = 0.01, w = 16):
    frame_list = sorted(list(poses_3D.keys()))
    frames = list(range(min(frame_list), max(frame_list) +1 ))

    data_p = []
    # Interpolate
    for i in range(len(frame_list)):
        f = frame_list[i]
        record = poses_3D[f]
        data_p.append(record["p3d"])

    data_p = np.array(data_p)
    interpolate_p = interpolate.interp1d(frame_list, data_p.T, kind = 'quadratic')
    data_p = interpolate_p(frames).T
    # Smooth
    for i in range(3):
        data_p[:,i] = filter1d(data_p[:,i], filter_t)
    
    # Velocity
    offset = data_p[1:] - data_p[:-1]
    velocity = np.linalg.norm(offset, axis = -1)
    mean_kernel = np.ones(w)/w
    velocity = np.convolve(velocity, mean_kernel, mode='same') 
    
    print(len(data_p), len(frame_list), frames[-1] - frames[0], frames[0], frames[-1], )
    last_p = data_p[0]
    poses_3D[frames[0]] = {"p3d":data_p[0], "offset": np.array([0.0,0.0,0.0]) }
    for i, f in enumerate(frames[1:]):
        if velocity[i] > theta_v: 
            poses_3D[f] = {"p3d":last_p + offset[i], "offset": offset[i]}
        else: poses_3D[f] = {"p3d":last_p, "offset": np.array([0.0,0.0,0.0])}
        last_p = copy.deepcopy(poses_3D[f]["p3d"] ) 
    return poses_3D

def main_3dproj_toys(info, datapath, timestamps,  output_file = "temp.npy"):
    cam_mat = np.load("camera_mat.npy", allow_pickle = True).item()
    cam_room = np.load("camera_room.npy", allow_pickle = True).item()

    width, height, starts, ends, cams = extract_video_info(info, timestamps)
    poses = read_toys(datapath)

    current_segment = 0
    curr_cam = cam_room
    if cams[current_segment] == 'c':
        curr_cam = cam_mat
    frame_list = list(poses.keys())
    sorted(frame_list)        
    
    toy_3d = {toy:{} for toy in CLASSES}
    for frame in frame_list:  
        if frame > ends[-1]: 
            print(frame)
            break
        
        if frame > ends[current_segment]:
            current_segment += 1
            curr_cam = cam_room
            if cams[current_segment] == 'c':
                curr_cam = cam_mat        
        if not starts[current_segment] < frame: continue
        
        # 3D Projection
        data_toys = to_3D_pose(poses[frame], curr_cam, height, width)
        for toy, info in data_toys.items():  
            if not "p3d" in info: continue
            toy_3d[toy][frame] = info
        if frame % 1000 == 0:
            print(current_segment, cams[current_segment], starts[current_segment], ends[current_segment], frame)

    """for t, data in toy_3d.items():
        if len(data.keys()) > 0:
            min_f = min(data.keys())
            print(t, data[min_f])"""

    for s, e in zip(starts, ends):
        for toy in CLASSES:
            p_temp = {k:v for k, v in toy_3d[toy].items() if s <= k <= e}
            if len(p_temp) == 0: continue
            frame_min, frame_max = min(toy_3d[toy].keys()), min(toy_3d[toy].keys())
            for i in range(s, frame_min): p_temp[i] = copy.deepcopy(toy_3d[toy][frame_min])
            #for i in range(frame_max, e): p_temp[i] = np.copy(toy_3d[toy][frame_max])
            p_temp = smooth_3D(p_temp, filter_t = 3)
            toy_3d[toy].update(p_temp)
            
    np.save(output_file, toy_3d)
    return 


def process_toys(video_info, path_processed, output_dir, whitelist=None):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for name, info in video_info.items():
        filename = os.path.join(path_processed, info["download"]["name"] + "_toys.npy")
        output_file = os.path.join(output_dir, info["download"]["name"] + "_toys3D.npy")
        if not os.path.exists(filename):
            continue       
        if not "download" in info: 
            print(name, " could not download")
            continue
        if whitelist is not None:
            if not name in whitelist: continue
        else:
            # Check if video has been already processed
            if os.path.exists(output_file): continue
                
        # Download video
        timestamps = {key: info[key] for key in ['c', 'r', 'p'] if key in info}
        if len(timestamps) == 0 : continue
        print(name, timestamps)
        print(info)
        #exit()
        main_3dproj_toys(info, filename, timestamps, output_file = output_file)
    return

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--toy_dir', type=str, default="/nfs/hpc/cn-gpu5/DevEv/viz_toys/", help="Directory path containing processed videos data of toy tracking.")
    parser.add_argument('--output_dir', type=str, default="/nfs/hpc/cn-gpu5/DevEv/viz_toys3D/", help="Directory path where toy pose files will be written")
    parser.add_argument('--timestamps', type=str, default="DevEvData_2024-02-02.csv", help="Path to timestamp file")
    parser.add_argument('--uname', type=str, default="", help="Databrary username")
    parser.add_argument('--psswd', type=str, default="", help="Databrary password")
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
    process_toys(video_info, args.input_dir, args.toy_dir, whitelist = args.session)