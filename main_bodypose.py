import os
from multiprocessing import Process

from databrary import get_videos, get_processed_body, download, get_remaining
from BodyPose import inferbody

#SUBLIST = ["16_04", "24_02", "26_06", "27_03", "27_06", "28_01", "29_02", "29_03", "29_07"]
#SUBLIST = ["29_01", "29_09", "33_01", "34_02", "34_03", "35_01"]]

def process_body(video_info, session, target_dir, path_processed, log_file = "processed_video_body.csv", whitelist = None):
    if not os.path.exists(log_file):
        with open(log_file, "w") as f:
            f.write("")

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    for name, info in video_info.items():
        #if not name in SUBLIST: continue
        
        # Select only first half of available files
        # if i < len(record)//2: continue For Tieqiao to uncomment
        if whitelist is not None:
            if not name in whitelist: continue
        if not "download" in info: 
            print(name, " could not download")
            continue

        # Check if video has been already processed
        processed = get_processed_body(soft=True, path=path_processed)
        if any(name in p for p in processed): continue
        
        # Download video
        video_output = os.path.join(target_dir, "{}.{}".format(info["download"]["name"], info["download"]["format"]))
        timestamps = {key: info[key] for key in ['c', 'r', 'p'] if key in info}
        print(name, timestamps)
        if not os.path.exists(video_output):
            print("Downloading {} in {}".format(info["download"]["name"], target_dir))
            download(info["download"]["url"], session, video_output)
            print("Download Completed")
        if not os.path.exists(video_output):
            print("Error during download for", video_output)
            continue
        # Infer Body Pose
        inferbody(video_output, timestamps, write=True, output_dir=path_processed)
        with open(log_file, "a") as f:
            f.write("{},finished\n".format(info["download"]["name"]))
    return


def check_body(path_processed):
    import re
    npz = [f for f in os.listdir(path_processed) if f.endswith("_Sync.npz")]
    txt = [f for f in os.listdir(path_processed) if f.endswith("_Sync.txt")]

    remaining, finished = [], []
    for f in txt:
        sess_name = re.findall(r'\d\d_\d\d', f)[0]
        if any(sess_name in s for s in npz): 
            finished.append(sess_name)
        else: 
            remaining.append(sess_name)

    return finished, remaining

def check_timestamps(video_info, path_processed):
    import cv2
    
    wrong_files = []
    for name, info in video_info.items():
        if not "download" in info: 
            print(name, " could not download")
            wrong_files.append(info["download"]["name"])
            continue

        bodytxt = os.path.join(path_processed, "output_" + info["download"]["name"] + ".txt")
        video_output = os.path.join(target_dir, "{}.{}".format(info["download"]["name"], info["download"]["format"]))
        vidcap = cv2.VideoCapture(video_output)
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        if not os.path.exists(bodytxt): 
            print(info["download"]["name"], "not found")
            wrong_files.append(info["download"]["name"])
            continue
        timestamps = {key: info[key] for key in ['c', 'r', 'p'] if key in info}
        
        with open(bodytxt, "r") as f:
            body = f.readlines()        
        frames = set([int(x.split(",")[0]) for x in body if len(x) > 0])
        
        for n, stamps in timestamps.items():
            for (s,e) in stamps:
                start, end = int(fps*s/1000), int(fps*e/1000)
                sublist_frames = sorted([f for f in frames if start<=f<=end])
                
                diff = 100*len(sublist_frames)/ (end-start+1)
                if diff < 80.0: 
                    wrong_files.append(info["download"]["name"])
                print(n, start, end, diff, len(sublist_frames))
    wrong_files = list(set(wrong_files))
    print(wrong_files)
    return wrong_files

if __name__ == "__main__":
    target_dir = "/nfs/hpc/cn-gpu5/DevEv/dataset/" # output dir to save videos
    path_processed = "/nfs/hpc/cn-gpu5/DevEv/viz_bodypose_new/"
    timestamps = "DevEvData_2023-05-16.csv"
    uname = 'azieren@oregonstate.edu'
    psswd = 'changetheworld38'
    
    if not os.path.exists(path_processed):
        os.makedirs(path_processed)
    #finished, remaining = check_body(path_processed)
    #print(remaining, len(finished), len(remaining))

    video_info, session = get_videos(uname, psswd, timestamps)
    #finished, remaining = get_remaining(video_info, path_processed, "body")
    
    #check_timestamps(video_info, path_processed)

    process_body(video_info, session, target_dir, path_processed, whitelist = None)




