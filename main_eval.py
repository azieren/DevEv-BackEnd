import os
import re
import numpy as np
from collections import OrderedDict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
from databrary import get_videos

def read_timestamp(filepath="/nfs/hpc/share/azieren/DevEv/DevEvData_2023-06-20.csv"):
    with open(filepath) as f:
        text = f.readlines()
    
    text = [l.split(",") for l in text[1:]]
    record = OrderedDict()
    for data in text:
        if data[1] not in record:
            # Processed flag: False means the the method has not been processed yet
            record[data[1]] = {}
        if len(data) <= 25: category = data[-3]
        else: category = data[-6]
        if category in ['c', 'r', 'p']:
            if len(data) <= 25:
                onset, offset = int(data[-2]), int(data[-1])
            else:
                onset, offset = 29.97*int(data[-5])/1000, 29.97*int(data[-4])/1000
            if category not in record[data[1]]: record[data[1]][category] = []
            record[data[1]][category].append((onset, offset))
            
    return record

def read_gt_mv(file_path, data_dir, view_mode = "room"):
    assert view_mode in ["mat", "room"]
    # input:    relative path to .txt file with file names
    # output:   list of relative path names
    MATVIEWS = ["17_03", "19_01", "19_02", "20_02", "20_03", "25_01", "27_01", "29_01", "33_01", "34_02", "34_03", "34_04", "35_01"]

    timestamps = read_timestamp()

    print(file_path)
    with open(file_path) as f:
        lines = f.read().splitlines()
    data, count = {}, 0
    for l in lines:
        if len(l) <= 0: continue
        info = l.split(",")
        if len(info) != 13: continue
        name, is_corrected, is_front, x1, y1, x2, y2, y, p, r, x3d, y3d, z3d = info
        bbox = [int(x) for x in [x1, y1, x2, y2]]
        p3d = np.array([float(x) for x in [x3d, y3d, z3d]])
        name = name.replace("S", "")
        sess, subj, frame, view = name.replace(".png", "").split("_")
        sess_name = "{}_{}".format(sess, subj)
        if sess_name not in timestamps: continue
        # Evaluation mode
        #if not (sess_name in TEST and train_mode): continue
        # Find which type of camera setup
        type_views = None
        for cam, segments in timestamps[sess_name].items():
            for  (start, end) in segments:
                if start <= int(frame) <= end:
                    type_views = cam
                    break
            if type_views is not None: break
        # Remove mat views
        #if type_views == "c" and view_mode == "room": continue
        #elif type_views != "c" and view_mode == "mat": continue
        
        path = os.path.join(data_dir, name)
        angles = np.array([float(y), float(p), float(r)])
        is_corrected = True if int(is_corrected) == 0 else False
        if not is_corrected: continue
        #if not os.path.exists(path) or not is_corrected: 
        #    continue
        view = int(view)
        if not sess_name in data: 
            data[sess_name] = {}
        if frame in data[sess_name]: continue    
        data[sess_name][int(frame)] = {"path":name, "is_front":int(is_front), "dir":angles, "type":type_views, "box":bbox, "p3d":p3d}

    return data
 
def read_prediction(filename):
    if not os.path.exists(filename): 
        return {}
    attention = {}

    with open(filename, "r") as f:
        data = f.readlines()

    for i, d in enumerate(data):
        d_split = d.replace("\n", "").split(",")
        xhl, yhl, zhl, xhr, yhr, zhr = 0,0,0,0,0,0
        flag, flag_h = 0, 0
        if len(d_split)== 10:
            frame, b0, b1, b2, A0, A1, A2, att0, att1, att2 = d_split
        elif len(d_split)== 11:
            frame, b0, b1, b2, A0, A1, A2, att0, att1, att2, flag = d_split
        elif len(d_split)== 18:
            frame, flag, flag_h, b0, b1, b2, A0, A1, A2, att0, att1, att2, xhl, yhl, zhl, xhr, yhr, zhr = d_split
        elif len(d_split) < 10: continue
        else:
            print("Error in attention file")
            exit()
        flag, flag_h = int(flag), int(flag_h)
        pos = np.array([float(att0), float(att1), float(att2)])
        #vec = np.array([float(A0), float(A1), float(A2)])
        b = np.array([float(b0), float(b1), float(b2)])
        handL = np.array([float(xhl), float(yhl), float(zhl)])
        handR = np.array([float(xhr), float(yhr), float(zhr)])
        vec = (pos - b)
        n = np.linalg.norm(vec)
        #if n < 1e-6:
        #    print("Invalid", int(frame))
        vec = vec / (n+1e-6)
        attention[int(frame)] = {"head":b, "att":vec, "handL":handL,"handR":handR, "flag":flag, "flag_h":flag_h}
            
    return attention

def read_gt_hands(corrected_path):
    case = os.path.join(corrected_path, "right")
    case = [os.path.join(case, x) for x in os.listdir(case) if x.endswith(".txt")]
    
    
    dataset = {}
    total = 0
    for filename in case:   
        data = read_prediction(filename)
        sess = filename.split("/")[-1]
        sess = re.findall(r'\d{2}_\d{2}', sess)[0]
        data = {k:{"handR":v["head"], "handL":None} for k, v in data.items() if v["flag"] == 1}
        dataset[sess] = data
        total += len(data)

    case = os.path.join(corrected_path, "left_right")
    case = [os.path.join(case, x) for x in os.listdir(case) if x.endswith(".txt")]
    for filename in case:   
        data = read_prediction(filename)
        sess_file = filename.split("/")[-1]
        sess = re.findall(r'\d{2}_\d{2}', sess_file)[0]
        hasRight, hasLeft = "Right" in sess_file, "Left" in sess_file

        if sess not in dataset: dataset[sess] = {}
        for k, v in data.items():
            if not v["flag_h"] == 1: continue
            if k not in dataset[sess]: dataset[sess][k] = {"handR":None, "handL":None}
            if hasRight: 
                dataset[sess][k]["handR"] = v["handR"]
                total += 1
            if hasLeft: 
                dataset[sess][k]["handL"] = v["handL"]
                total += 1
   
    return dataset, total
    
def get_angular_error(gt, pred):
    similarity = pred*gt
    return np.arccos(similarity)*180/np.pi
        
def eval_head(video_info, session, target_dir, path_processed):

    data = read_gt_mv("/nfs/hpc/cn-gpu5/DevEv/headpose_dataset/gt_body.txt", "", view_mode = "room")
    error_head_pos, error_head_dir = [], []
    total = 0
    for sess_name, sess_info in data.items():
        total += len(sess_info.keys())
        
        data_session = video_info[sess_name]
        data_3d = read_prediction(os.path.join(path_processed, "attC_{}.txt".format(data_session["download"]["name"])))
        if len(data_3d) == 0: print("Missing", sess_name)
        for frame, info_frame in sess_info.items():
            if not frame in data_3d: continue
            head_pos_gt, head_pos_pred = info_frame["p3d"], data_3d[frame]["head"]
            head_dir_gt, head_dir_pred = info_frame["dir"], data_3d[frame]["att"]
            
            err_pos = np.linalg.norm(head_pos_gt - head_pos_pred)
            err_dir = get_angular_error(head_dir_gt, head_dir_pred )
            error_head_pos.append(err_pos)
            error_head_dir.append(err_dir)
    error_head_pos = np.array(error_head_pos)   
    
    mu, sigma = np.mean(error_head_pos), np.std(error_head_pos)    
    print("Total Dataset {} / Total Considered {}".format(total, len(error_head_pos)))
    print("Test Error Head Position (m): {:.3f} +/- {:.3f}".format(np.mean(error_head_pos), np.std(error_head_pos)))  
    print("Test Error Head Orientation (deg): {:.3f} +/- {:.3f}".format(mu, sigma))    
    
    print(min(error_head_pos), max(error_head_pos))
    # Define bins and compute histogram
    error_head_pos[error_head_pos >= 1] = 1.0
    bins = np.arange(0.0, max(error_head_pos) + 0.05, 0.05)
    hist, bin_edges = np.histogram(error_head_pos, bins=bins)
    # Normalize by total area (bin width)
    hist_normalized = hist / len(error_head_pos)
    print(hist_normalized)
    # Plot the normalized histogram
    plt.bar(bin_edges[:-1] + 0.025, hist_normalized, width=0.05, edgecolor='black', alpha=0.75)
    plt.xlabel('Error Head (m)')
    plt.ylabel('Density')
    plt.xticks(bin_edges, rotation=45, ha="right")
    plt.yticks(np.arange(0, max(hist_normalized) + 0.05, 0.1))
    plt.title("Error Head Hist - {:.3f} +/- {:.3f}".format(mu, sigma))
    plt.tight_layout()
    plt.savefig("eval/error_hist_head.png")   
    plt.close()
    return

def eval_hands(video_info, session, target_dir, path_processed):
    data, total = read_gt_hands("corrected_hands/")
    error_hand = []

    for sess_name, sess_info in data.items():     
        data_session = video_info[sess_name]
        data_3d = read_prediction(os.path.join(path_processed, "attC_{}.txt".format(data_session["download"]["name"])))
        if len(data_3d) == 0: print("Missing", sess_name)
        print(sess_name, len(sess_info))
        for frame, info_frame in sess_info.items():
            if not frame in data_3d: 
                print("Frame not found", sess_name, frame)
                continue
            head_HL_gt, head_HL_pred = info_frame["handL"], data_3d[frame]["handL"]
            head_HR_gt, head_HR_pred = info_frame["handR"], data_3d[frame]["handR"]
            
            if head_HL_gt is not None:
                err_pos = np.linalg.norm(head_HL_gt - head_HL_pred)
                error_hand.append(err_pos)
            if head_HR_gt is not None:
                err_pos = np.linalg.norm(head_HR_gt - head_HR_pred)
                error_hand.append(err_pos)
    error_hand = np.array(error_hand)            
      
    mu, sigma = np.mean(error_hand), np.std(error_hand)
    print("Total Dataset {} / Total Considered {}".format(total, len(error_hand)))
    print("Test Error Hand Position (m): {:.3f} +/- {:.3f}".format(mu, sigma))  
  
    print(min(error_hand), max(error_hand))
    # Define bins and compute histogram
    error_hand[error_hand >= 1] = 1.0
    bins = np.arange(0.0, max(error_hand) + 0.05, 0.05)
    hist, bin_edges = np.histogram(error_hand, bins=bins)
    # Normalize by total area (bin width)
    hist_normalized = hist / len(error_hand)
    print(hist_normalized)
    # Plot the normalized histogram
    plt.bar(bin_edges[:-1] + 0.025, hist_normalized, width=0.05, edgecolor='black', alpha=0.75)
    plt.xlabel('Error Hand (m)')
    plt.ylabel('Density')
    plt.xticks(bin_edges, rotation=45, ha="right")
    plt.yticks(np.arange(0, max(hist_normalized) + 0.05, 0.1))
    plt.title("Error Hand Hist - {:.3f} +/- {:.3f}".format(mu, sigma))
    plt.tight_layout() 
    plt.savefig("eval/error_hist_hand.png")   
    plt.close()
    return

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default="/nfs/hpc/cn-gpu5/DevEv/dataset/", help="Directory path where 3D pose files will be written")
    parser.add_argument('--head_dir', type=str, default="/nfs/hpc/cn-gpu5/DevEv/viz_attention/", help="Directory path where head pose files are")
    parser.add_argument('--timestamps', type=str, default="DevEvData_2024-02-02.csv", help="Path to timestamp file")
    parser.add_argument('--uname', type=str, default="azieren@oregonstate.edu", help="Databrary username")
    parser.add_argument('--psswd', type=str, default="changetheworld38", help="Databrary password")
    parser.add_argument('--check_time', action="store_true", help="Used only for checking the current amount of frames processed by existing files")
    parser.add_argument('--video_dir', type=str, default="/nfs/hpc/cn-gpu5/DevEv/dataset/", help="Directory path containing original videos. Only used with --check_time")
    args = parser.parse_args()
    
    
    if args.uname == "":
        print("Enter a Databrary Username")
        exit()
    if args.uname == "":
        print("Enter a Databrary Password")
        exit()
    return args


if __name__ == "__main__":
    args = parse_args()
    video_info, session = get_videos(args.uname, args.psswd, args.timestamps)
    
    eval_head(video_info, session, args.input_dir, args.head_dir)
    eval_hands(video_info, session, args.input_dir, args.head_dir)

