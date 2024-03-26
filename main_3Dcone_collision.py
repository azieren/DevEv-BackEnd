import os
import re
import numpy as np
import argparse
import trimesh

from databrary import get_processed_cone

TOY_MAPPING = {
    'pink_ballon':'pink_ball', 'tower_bloc':'red_tower', 'cylinder_tower':'tower', 'ball_container':'bucket',
}

def compute_bounding_box_center(vertices):
    # Convert the list of vertices to a NumPy array for easier manipulation
    vertices_array = np.array(vertices)

    # Compute the minimum and maximum coordinates along each axis (x, y, z)
    min_coords = np.min(vertices_array, axis=0)
    max_coords = np.max(vertices_array, axis=0)

    # Calculate the center point by averaging the minimum and maximum coordinates along each axis
    bounding_box_center = (min_coords + max_coords) / 2
    return bounding_box_center

class OBJ:
    def __init__(self, filename, swapyz=False):
        dirname = os.path.dirname(filename)
        """Loads a Wavefront OBJ file. """
        vertices = []
        normals = []
        textures = []
        count = 0
        self.content = {}

        for line in open(filename, "r"):
            if line.startswith('#'): continue
            values = line.split()
            if not values: continue
            if values[0] == 'o':
                curr = values[1]
                self.content[curr] = {"vertexes":[], "textures":[], "faces":[], "normals":[], "material":[], "count":[]}
                count = 0
            if values[0] == 'v':
                #v = map(float, values[1:4])
                v = [float(x) for x in values[1:4]]
                if swapyz:
                    v = v[0], -v[2], v[1]
                vertices.append(v)
            elif values[0] == 'vn':
                #v = map(float, values[1:4])
                v = [float(x) for x in values[1:4]]
                if swapyz:
                    v = v[0], -v[2], v[1]
                normals.append(v)
            elif values[0] == 'vt':
                vt = [float(x) for x in values[1:3]]
                #vt = 1-vt[1], vt[0]
                textures.append(vt)
            elif values[0] in ('usemtl', 'usemat'):
                self.content[curr]["material"].append(values[1])
                self.content[curr]["count"].append(len(self.content[curr]["vertexes"]))
            elif values[0] == 'mtllib':
                self.mtl = os.path.join(dirname ,values[1])
            elif values[0] == 'f':
                face = []
                texcoords_ = []
                norms = []
                for v in values[1:]:
                    w = v.split('/')
                    face.append(int(w[0]))
                    if len(w) >= 2 and len(w[1]) > 0:
                        texcoords_.append(int(w[1]))
                    else:
                        texcoords_.append(0)
                    if len(w) >= 3 and len(w[2]) > 0:
                        norms.append(int(w[2]))
                    else:
                        norms.append(0)                
                self.content[curr]["vertexes"].extend([vertices[x-1]  for x in face])
                self.content[curr]["textures"].extend([textures[x-1] for x in texcoords_])                
                self.content[curr]["normals"].extend([normals[x-1]  for x in norms])
                self.content[curr]["faces"].extend([count, count+1, count +2])
                count += 3
        return

def read_room(obj_file):
    obj = OBJ(obj_file, swapyz=True)

    vertices = []
    faces = []
    normals = []
    face_names = []
    toy_objects = {}
    count = 0
    for name, ob in obj.content.items():
        if len(ob["material"]) == 0: 
            print(name, np.array(ob["vertexes"]).shape)
            continue
        if "camera" in name: continue
        
        vert = np.array(ob["vertexes"]).reshape(-1, 3)
        if len(vert) == 0: continue
        face = np.array(ob["faces"]).reshape(-1, 3)
        normal = np.array(ob["normals"]).reshape(-1, 3)
        face_name = np.repeat(name, len(face), axis=0)
    
        if "toy" in name:        
            center = compute_bounding_box_center(vert)
            toy_objects[name.replace("toy_", "")] = {"vert":vert, "faces":face, "face_names":face_name,
                                                            "normals":normal, "center":center, "data":{}}
            continue
        
        vertices.append(vert)
        normals.append(normal)
        faces.append(face + count)
        face_names.append(face_name)
        count += face[-1][-1] + 1
        
    vertices = np.concatenate(vertices, axis = 0)
    faces = np.concatenate(faces, axis = 0) 
    face_names = np.concatenate(face_names, axis = 0) 
    normals = np.concatenate(normals, axis = 0)  
    room = {"vert":vertices, "faces":faces, "normals":normals, "face_names":face_names}
    return room, toy_objects

def generate_uniform_lines_in_cone(point, direction, angle_degrees, N):
    angle = np.radians(angle_degrees)  # Convert angle from degrees to radians
    direction = direction /np.linalg.norm(direction)

    cos_theta_max = np.cos(angle)
    cos_theta_values = np.linspace(cos_theta_max, 1.0, num=N)
    theta_values = np.arccos(cos_theta_values)
    phi_values = np.random.uniform(0, 2 * np.pi, size=N)

    x_values = np.sin(theta_values) * np.cos(phi_values)
    y_values = np.sin(theta_values) * np.sin(phi_values)
    z_values = np.cos(theta_values)

    local_coordinates = np.vstack((x_values, y_values, z_values)).T
    rotation_matrix = compute_rotation_matrix(direction)
    global_coordinates = np.dot(local_coordinates, rotation_matrix.T)

    return global_coordinates

def generate_lines_from_points(point, points):
    lines = []
    for p in points:
        lines.append([point, p])
    return lines

def compute_rotation_matrix(direction):
    v = np.array([direction[1], -direction[0], 0])
    if np.linalg.norm(v) == 0:
        v = np.array([0, direction[2], -direction[1]])

    u = np.cross(direction, v)
    u = u /np.linalg.norm(u)

    rotation_matrix = np.vstack((u, np.cross(direction, u), direction))

    return rotation_matrix

def find_intersections_with_mesh(start_point, ray_directions, mesh):

    # Repeat start point to match the number of ray directions
    ray_origins = np.repeat(start_point.reshape(1,3), len(ray_directions), axis=0)

    # Perform ray-mesh intersection
    intersections, index_ray, index_tri = mesh.ray.intersects_location(
        ray_origins=ray_origins, ray_directions=ray_directions, multiple_hits=False)
    
    if len(intersections) == 0: return [], [], []
    
    # Map the triangle index to the corresponding face name
    face_name = [mesh.metadata['face_names'][tri] for tri in index_tri]
    
    # Return the intersected triangle index and the corresponding face name
    return intersections, face_name, index_ray

def read_attention(filename):
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
        attention[int(frame)] = {"head":b, "att":vec, "handL":handL,"handR":handR}
            
    return attention

def main_3dcone(video_info, cone_angle, n_lines, output_dir = "output_3Dcone/"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(output_dir, video_info["name"] + "_cone3D.txt")
    with open(output_file, "w") as f:
        f.write("#Session name, cone angle, number of line per cone\n")
        f.write("#{},{},{}\n".format(video_info["name"], cone_angle, n_lines))
        f.write("#frame, line id, object name, x_att, y_att, z_att, x_h, y_h, z_h\n")
    attention = read_attention(video_info["att"])
    toys_info = np.load(video_info["toy"], allow_pickle=True).item() 
    
    frame_list = list(attention.keys())
    frame_list = sorted(frame_list)        

    room_mesh, toy_objects = read_room("Room.obj")


    # Create a Toy meshes
    for n, info in toy_objects.items():
        toy_objects[n]["mesh"] = trimesh.Trimesh(vertices=info['vert'], faces=info['faces'], vertex_normals=info['normals'])
        toy_objects[n]["mesh"].metadata['face_names'] = info['face_names']
        name = n
        if n in TOY_MAPPING: name = TOY_MAPPING[n]
        if not name in toys_info: continue
        toy_objects[n]["data"] = toys_info[name]
    # Convert mesh data to trimesh objects
    room_trimesh = trimesh.Trimesh(vertices=room_mesh['vert'], faces=room_mesh['faces'], vertex_normals=room_mesh['normals'])
    room_trimesh.metadata['face_names'] = room_mesh['face_names']
    
    
    for count, frame in enumerate(frame_list):  
        att_info = attention[frame]
        vecs = generate_uniform_lines_in_cone(att_info["head"], att_info["att"], cone_angle, n_lines)
        intersection_points, intersection_objects, index_ray = find_intersections_with_mesh(att_info["head"], vecs, room_trimesh)
        
        collision3D = {i:{} for i in range(n_lines) }
        for ray, p, obj in zip(index_ray, intersection_points, intersection_objects):
            collision3D[ray] = {"p":p.reshape(3), "name":obj}
        
        for n, info in toy_objects.items():
            translation_vector = np.array([0.0,0.0,0.0])
            if frame in info["data"] and "p3d" in info["data"][frame]:
                translation_vector = info["data"][frame]["p3d"].reshape(3) - info["center"]
            info["mesh"].vertices += translation_vector.reshape(1,3)
            intersection_points, intersection_objects, index_ray = find_intersections_with_mesh(att_info["head"], vecs, info["mesh"])
            info["mesh"].vertices -= translation_vector.reshape(1,3)
            for ray, p, obj in zip(index_ray, intersection_points, intersection_objects):
                if not "p" in collision3D[ray]: 
                    collision3D[ray] = {"p":p.reshape(3), "name":obj}
                    continue
                p0 = np.linalg.norm(att_info["head"] - collision3D[i]["p"])
                p1 = np.linalg.norm(att_info["head"] - p)
                if p1 < p0:
                    collision3D[ray] = {"p":p, "name":obj}
        
            
        xh, yh, zh = att_info["head"]
        with open(output_file, "a") as f:
            for i in range(n_lines):
                if len(collision3D[i]) == 0: continue
                x, y, z = collision3D[i]["p"]
                name = collision3D[i]["name"]
                f.write("{:d},{:d},{},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}\n".format(frame, i, name, x, y, z, xh, yh, zh))
        if count % 200 == 0:
            print("{}/{}".format(count, len(frame_list)))
        #if count > 1000: break
    print("{} finished and saved to {}".format(video_info["name"], output_file))
    return 

def process_3Dcone(video_info, path_processed, cone_angle, n_lines, whitelist=None):
    if not os.path.exists(path_processed):
        os.makedirs(path_processed)

    for name, info in video_info.items():
        # Select only first half of available files
        #if not name in SUBLIST: continue
        if not ("att" in info and "toy" in info): continue
        if not (info["att"] and info["toy"]) : continue
        if whitelist is not None:
            if not name in whitelist: continue
        else:
            # Check if video has been already processed
            processed = get_processed_cone(path=path_processed)
            if any(name in p for p in processed): continue
        print(name)
        print(info)
        info["name"] = name
        main_3dcone(info, cone_angle, n_lines, output_dir = path_processed)
    return
     
def get_att_toy_files(video_info, path_att, path_toys):
    video_info = {}
    for filename in os.listdir(path_att):
        if not filename.endswith(".txt"): continue
        sess_name = re.findall(r'\d\d_\d\d', filename)
        if len(sess_name) == 0: continue
        if not sess_name in video_info: video_info[sess_name] = {}
        video_info[sess_name]["att"] = os.path.join(path_att, filename)
    
    for filename in os.listdir(path_toys):
        if not filename.endswith(".npy"): continue
        sess_name = re.findall(r'\d\d_\d\d', filename)
        if len(sess_name) == 0: continue
        if not sess_name in video_info: video_info[sess_name] = {}
        video_info[sess_name]["toy"] = os.path.join(path_toys, filename)

    return video_info

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default="/nfs/hpc/cn-gpu5/DevEv/viz_cone3D/", help="Directory path where 3D pose files will be written")
    parser.add_argument('--att_dir', type=str, default="/nfs/hpc/cn-gpu5/DevEv/viz_attention/", help="Directory path where attention files are")
    parser.add_argument('--cone_angle', type=float, default=20.0, help="Angle of the cone to generate lines from (0 to 90 degrees)")
    parser.add_argument('--n_lines', type=int, default = 10, help="Number of ray to cast in the cone")
    parser.add_argument('--toy_dir', type=str, default="/nfs/hpc/cn-gpu5/DevEv/viz_toys3D/", help="Directory path where 3D toys pose files are")
    parser.add_argument('--session', default = "", type=str, help="If used, only this session will be processed. Format: session and subject number ##_##")
    
    args = parser.parse_args()
    sess_name = re.findall(r'\d\d_\d\d', args.session)
    if len(sess_name) == 0:
        args.session = None
    else:
        args.session = sess_name[0]
    assert 0 < args.cone_angle <= 90
    assert 1 <= args.n_lines
    return args

if __name__ == "__main__":
    args = parse_args()
    video_info = get_att_toy_files(args.att_dir, args.toy_dir) 
    process_3Dcone(video_info, args.output_dir, args.cone_angle, args.n_lines, whitelist = args.session)
    