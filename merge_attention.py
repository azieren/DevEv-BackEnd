import os
import numpy as np
import argparse
import trimesh

def read_attention(filename):
    if not os.path.exists(filename): 
        print("Attention file does not exists", filename)
        exit()
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

        size = np.linalg.norm(pos - b)
        if size < 1e-6: 
            attention[int(frame)] = np.copy(attention[int(frame) - 1]).item()
            continue
        vec = (pos - b)/ ( size + 1e-6)

        attention[int(frame)] = {"orientation":vec, "head":b, "att":pos, "corrected_flag":flag,
                                 "corrected_flag_hand":flag_h, "handL":handL,"handR":handR}
    
    return attention

def write_attention(poses_3D, output_file):
    count = 0
    frames = np.sort(list(poses_3D.keys()))
    att = None
    mesh = trimesh.load_mesh('Room.ply')

    with open(output_file, "w") as f:

        for frame in frames:
            record = poses_3D[frame]
            b, A = record["head"], record["orientation"]
            handL, handR = record["handL"], record["handR"]
            flag1, flag2 = record["corrected_flag"], record["corrected_flag_hand"]
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
                frame, flag1, flag2, b[0], b[1], b[2], A[0], A[1], A[2], att[0], att[1], att[2], handL[0], handL[1], handL[2], handR[0], handR[1], handR[2]
                ))
            count += 1
            if count % 500 == 0:
                print("writing", count)

        return
    
def process_attention(head_pos, head_or, output_file):
    head_pos_info = read_attention(head_pos)
    head_or_info = read_attention(head_or)
    
    for f, info in head_pos_info.items():
        if not f in head_or_info: continue
        info["orientation"] = head_or_info[f]["orientation"]
        
    write_attention(head_pos_info, output_file)  
    return

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_file', type=str, default="/nfs/hpc/cn-gpu5/DevEv/viz_attention/file_merged.txt", help="Directory path where 3D pose files will be written")
    parser.add_argument('--head_pos', type=str, default="/nfs/hpc/cn-gpu5/DevEv/viz_headpose/file1.txt", help="Attention files with head and hand position to copy from")
    parser.add_argument('--head_or', type=str, default="/nfs/hpc/cn-gpu5/DevEv/viz_headpose/file2.txt", help="Attention files with head orientation to copy from")
    args = parser.parse_args()
    return args



if __name__ == "__main__":
    args = parse_args()
    process_attention(args.head_pos, args.head_or, args.output_file)
