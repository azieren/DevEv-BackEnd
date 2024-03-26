import os
import re
import sys
import argparse

from databrary import get_videos, get_processed_toy, download, get_remaining
from toyTracking.inference_video import main_inference_toy

SUBLIST = ["05_04", "07_04","08_03", "13_03", "13_04", "14_03", "14_06", "15_03", "15_04", "15_09", 
           "15_10", "15_11", "16_04", "16_06", "18_01","18_05", "19_01", "19_02", "19_03", "19_04", 
           "19_06", "20_02", "20_03", "20_05", "20_06", "20_09", "21_03", "21_06", "21_08", 
           "24_02", "25_01", "26_02", "26_05", "26_06", "27_01", "27_02", "27_03", "27_06", 
           "28_01", "29_01", "29_02", "29_03", "29_07", "33_01", "34_02", "34_03", "34_04", "35_01"]
SUBLIST = ["05_04", "07_04","08_03", "13_03", "15_03", "15_04", "15_10"]

def process_toys(video_info, session, target_dir, path_processed, whitelist = None, write = False):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    if not os.path.exists(path_processed):
        os.makedirs(path_processed)
        
    for name, info in video_info.items():
        # Select only first half of available files
        if whitelist is not None:
            if not name in whitelist: continue
        else:
            # Check if video has been already processed
            processed = get_processed_toy(path=path_processed)
            if any(name in p for p in processed): continue
       
        if not "download" in info: continue
        
        # Download video
        video_output = os.path.join(target_dir, "{}.{}".format(info["download"]["name"], info["download"]["format"]))
        timestamps = {key: info[key] for key in ['c', 'r', 'p'] if key in info}
        if len(timestamps) == 0 : continue
        # Skip mat views
        #if 'c' in  timestamps: continue
        print(name, timestamps)
        if not os.path.exists(video_output):
            print("Downloading {} in {}".format(info["download"]["name"], target_dir))
            download(info["download"]["url"], session, video_output)
            print("Download Completed")
        # Track toys
        main_inference_toy(video_output, timestamps, write=write, output_folder=path_processed)

    return

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default="/nfs/hpc/cn-gpu5/DevEv/dataset/", help="Directory path containing original videos.")
    parser.add_argument('--output_dir', type=str, default="/nfs/hpc/cn-gpu5/DevEv/viz_toys/", help="Directory path where toy pose files will be written")
    parser.add_argument('--timestamps', type=str, default="DevEvData_2024-02-02.csv", help="Path to timestamp file")
    parser.add_argument('--uname', type=str, default="", help="Databrary username")
    parser.add_argument('--psswd', type=str, default="", help="Databrary password")
    parser.add_argument('--write', action="store_true", help="If set, a video will be generated alongside the toy file")
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
    process_toys(video_info, session, args.input_dir, args.output_dir, whitelist = args.session, write = args.write)


