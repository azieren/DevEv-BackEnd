import os
from collections import OrderedDict
import re

from pybrary import Pybrary

SUPPORTED_FORMATS = {
    2: "csv",
    4: "rtf",
    5: "png",
    6: "pdf",
    7: "doc",
    8: "odf",
    9: "docx",
    10: "xls",
    11: "ods",
    12: "xlsx",
    13: "ppt",
    14: "odp",
    15: "pptx",
    16: "opf",
    18: "webm",
    20: "mov",
    -800: "mp4",
    22: "avi",
    23: "sav",
    24: "wav",
    19: "mpeg",
    26: "chat",
    -700: "jpeg",
    21: "mts",
    -600: "mp3",
    27: "aac",
    28: "wma",
    25: "wmv",
    29: "its",
    30: "dv",
    1: "txt",
    31: "etf"
}
def get_videos(uname, psswd, filename):
    # Get timestamped videos
    record = get_timestamp(filename)
    # Get already processed videos
    # Get url information for download
    sync_list, pb = get_databrary_videos(uname, psswd)
    # Include url information to timestamp
    record = fuse_timestamp_url(sync_list, record)
    return  record, pb

def get_remaining(video_info, path_processed, flag):
    if flag == "head":
        processed = get_processed_head(path=path_processed)
    else:
        processed = get_processed_body(soft=False, path=path_processed)
    
    video_info = get_to_do_videos(video_info, processed, flag)

    finished, remaining = [], []
    for name, info in video_info.items():
        if info[flag]: finished.append(name)
        else: remaining.append(name)
    return finished, remaining
    
def get_databrary_videos(uname, psswd):
    pb = Pybrary.get_instance(uname, psswd)
    volume_id = 1020 # DevEv volume id
    sync_list = retrieve_sync_videos(pb, volume_id)
    return sync_list, pb

def download(url, session, output):
    request = session.get_request_session()
    response = request.get(url, stream = True) 
    if response.status_code == 200:
        filepath = os.path.join(output)
        with open(filepath, 'wb') as f: 
            for chunk in response.iter_content(chunk_size = 1024*1024): 
                if chunk: 
                    f.write(chunk) 
        return True
    return False

def retrieve_sync_videos(session, volume_id):
    sync_list = []
    
    sess = [s for s in session.get_sessions(volume_id) if "name" in s]
    for s in sess:
        assets = session.get_session_assets(volume_id, s['id'])
        record = OrderedDict()
        for a in assets:
            a["name"] = a["name"].replace(".mp4","")
            if not a["name"].endswith("_Sync"): continue
            if not a["name"].startswith("DevEv_S"): continue
            format = SUPPORTED_FORMATS[a["format"]]
            assets_url = "/".join(["", "slot", str(s["id"]) , "-" , "asset" , str(a["id"]), "download"])
            video_url = "https://nyu.databrary.org{}".format(assets_url)
            sess_name = re.findall(r'\d\d_\d\d', a["name"])
            if len(sess_name) != 1: continue
            record = {"url":video_url, "name":a["name"], "id":str(s["id"]), "format":format, "session":sess_name[0]}
        if "url" in record: sync_list.append(record)
    return sync_list

def get_timestamp(filename="DevEvData_2023-06-20.csv"):
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
            record[data[1]][category].append((onset, offset))
            
    return record

def get_processed_head(path = "HeadPose/output"):
    # soft = True  reprocess unfinished files
    # soft = False remove fully processed files
    end = ".txt"
    return [ f for f in os.listdir(path) if f.endswith(end) ]

def get_processed_toy(path = "HeadPose/output"):
    # soft = True  reprocess unfinished files
    # soft = False remove fully processed files
    end = ".npy"
    return [ f for f in os.listdir(path) if f.endswith(end) ]


def get_processed_body(path = "BodyPose/output", soft = True):
    # soft = True  reprocess unfinished files
    # soft = False remove fully processed files
    end = ".npz"
    if soft: end = ".txt"
    return [ f for f in os.listdir(path) if f.endswith(end) ]

def get_to_do_videos(todofiles, donefile, flag):
    for f, info in todofiles.items():
        if any(f in s for s in donefile):
            info[flag] = True
    return todofiles

def fuse_timestamp_url(url_list, tsm_dict):
    for n, info in tsm_dict.items():
        for urlinfo in url_list:
            if n in urlinfo["name"]:
                info["download"] = urlinfo
                break

    return tsm_dict