import os
import re
import numpy as np
import argparse

from databrary import get_videos, get_processed_head, download, get_remaining
from HeadPose import inferheadMV2

MATVIEWS = ["17_03", "19_01", "19_02", "20_02", "20_03", "25_01", "27_01", "29_01", "33_01", "34_02", 
            "34_03", "34_04", "35_01"]

SUBLIST = ["05_04", "07_04","08_03", "13_03", "13_04", "14_03", "14_06", "15_03", "15_04", "15_09", 
           "15_10", "15_11", "17_03", "16_04", "16_06", "18_01","18_05", "19_01", "19_02", "19_03", "19_04", 
           "19_06", "20_02", "20_03", "20_05", "20_06", "20_09", "21_03", "21_06", "21_08", 
           "24_02", "25_01", "26_02", "26_05", "26_06", "27_01", "27_02", "27_03", "27_06", 
           "28_01", "29_01", "29_02", "29_03", "29_07", "29_09", "33_01", "34_02", "34_03", "34_04", "35_01"]

#SUBLIST = ["05_04", "07_04","08_03", "13_03"]

def process_head(video_info, session, target_dir, path_processed, body_dir, whitelist = None, write = False):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    if not os.path.exists(path_processed):
        os.makedirs(path_processed)
        
    for name, info in video_info.items():
        if not "download" in info: 
            print("No downlable video available")
            continue
        # Select only first half of available files
        if whitelist is not None:
            if not name in whitelist: continue
        else:
            # Check if video has been already processed
            processed = get_processed_head(path=path_processed)
            if any(name in p for p in processed): continue
            
        # Download video
        video_output = os.path.join(target_dir, "{}.{}".format(info["download"]["name"], info["download"]["format"]))
        timestamps = {key: info[key] for key in ['c', 'r', 'p'] if key in info}
        if len(timestamps) == 0: 
            print("No timestamps available")
            continue
        # Skip mat views
        #if 'c' in  timestamps: continue
        print(name, timestamps)
        if not os.path.exists(video_output):
            print("Downloading {} in {}".format(info["download"]["name"], target_dir))
            download(info["download"]["url"], session, video_output)
            print("Download Completed")
        # Infer Body Pose
        #inferheadMV(video_output, timestamps, write=False, output_folder=path_processed)
        inferheadMV2(video_output, timestamps, write=write, output_folder=path_processed, bbox_path=body_dir)
    return


def check_head(path_processed, path_body = "/nfs/hpc/cn-gpu5/DevEv/viz_bodypose"):
    txt = [f for f in os.listdir(path_processed) if f.endswith("_Sync.txt")] 

    finished, remaining = [], []
    for f in txt:
        sess_name = re.findall(r'\d\d_\d\d', f)[0]
        filename = f.replace("head-", "").replace(".txt", "")
        bodyfile = os.path.join(path_body, "data_2d_{}.npz".format(filename))
        if not os.path.exists(bodyfile): 
            remaining.append(sess_name)
            continue
        done = check_frames(bodyfile, os.path.join(path_processed, f))
        
        if not done:
            remaining.append(sess_name)
        else:
            finished.append(sess_name)
    return finished, remaining

def check_frames(bodyfile, headfile):
    
    with open(headfile, "r") as f:
        head = f.readlines()
    headframes = set([int(x.split(",")[0]) for x in head if len(x) > 0])
    headframes = sorted(list(headframes))

    body = np.load(bodyfile, allow_pickle=True)["data"].item()
    
    bodyframes = sorted(list([k for k, v in body.items() if len(v["views"]) > 0 ]))
    
    if headframes == bodyframes:
        return True
    inter =  sorted(list(set(bodyframes) - set(headframes)))
    return False

def check_timestamps(video_info, target_dir, path_processed):
    import cv2
    
    wrong_files = []
    for name, info in video_info.items():
        if not "download" in info: 
            print(name, " could not download")
            wrong_files.append((name, " could not download"))
            continue

        bodytxt = os.path.join(path_processed, "head-" + info["download"]["name"] + ".txt")
        video_output = os.path.join(target_dir, "{}.{}".format(info["download"]["name"], info["download"]["format"]))
        vidcap = cv2.VideoCapture(video_output)
        fps = vidcap.get(cv2.CAP_PROP_FPS)

        timestamps = {key: info[key] for key in ['c', 'r', 'p'] if key in info}
        if len(timestamps) == 0 : 
            wrong_files.append((name, "no timestamps"))
            continue
        if not os.path.exists(bodytxt): 
            wrong_files.append((name, "no head file"))
            continue
        with open(bodytxt, "r") as f:
            body = f.readlines()        
        frames = set([int(x.split(",")[0]) for x in body if len(x.replace("\n","")) > 0])
        
        for n, stamps in timestamps.items():
            for (s,e) in stamps:
                start, end = int(fps*s/1000), int(fps*e/1000)
                if end - start < 10: continue
                sublist_frames = sorted([f for f in frames if start<=f<=end])
                
                diff = 100*len(sublist_frames)/ (end-start+1)
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
    parser.add_argument('--input_dir', type=str, default="/nfs/hpc/cn-gpu5/DevEv/dataset/", help="Directory path containing original videos.")
    parser.add_argument('--output_dir', type=str, default="/nfs/hpc/cn-gpu5/DevEv/viz_headpose/", help="Directory path where head pose files will be written")
    parser.add_argument('--body_dir', type=str, default="/nfs/hpc/cn-gpu5/DevEv/viz_bodypose/", help="Directory path where body pose files are")
    parser.add_argument('--timestamps', type=str, default="DevEvData_2024-02-02.csv", help="Path to timestamp file")
    parser.add_argument('--uname', type=str, default="", help="Databrary username")
    parser.add_argument('--psswd', type=str, default="", help="Databrary password")
    parser.add_argument('--write', action="store_true", help="If set, a video will be generated alongside the headpose file")
    parser.add_argument('--check_time', action="store_true", help="Used only for checking the current amount of frames processed by existing files")
    parser.add_argument('--check_remaining', action="store_true", help="Used only for checking the missing files")
    parser.add_argument('--session', default = "", type=str, help="If used, only this session will be processed. Format: session and subject number ##_##")
    args = parser.parse_args()
    
    if args.uname == "":
        print("Enter a Databrary Username")
        exit()
    if args.psswd == "":
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
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    video_info, session = get_videos(args.uname, args.psswd, args.timestamps)

    if args.check_time:
        check_timestamps(video_info, args.input_dir, args.output_dir)
        exit()
    if args.check_remaining:
        finished, remaining = get_remaining(video_info, args.output_dir, "head")
        print("Finished videos: {}, Remaining videos {}".format(len(finished), len(remaining)))
        print("Remaining:", remaining)
        exit()
    process_head(video_info, session, args.input_dir, args.output_dir, args.body_dir, whitelist = args.session, write = args.write)

