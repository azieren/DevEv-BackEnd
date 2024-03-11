import os
import numpy as np
<<<<<<< HEAD

from collections import OrderedDict
import torch
from torch.utils.data.dataset import Dataset
=======
import cv2
import pandas as pd

import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms

>>>>>>> origin/main
from PIL import Image, ImageFilter
import utils



def get_list_from_filenames(file_path):
    # input:    relative path to .txt file with file names
    # output:   list of relative path names
    print(file_path)
    with open(file_path) as f:
        lines = f.read().splitlines()
    return lines

<<<<<<< HEAD
def read_gt(file_path, data_dir, train_mode):
=======
def read_gt(file_path):
>>>>>>> origin/main
    # input:    relative path to .txt file with file names
    # output:   list of relative path names
    print(file_path)
    with open(file_path) as f:
        lines = f.read().splitlines()
    
    data = {'image':[], 'pose':[]}
    for l in lines:
<<<<<<< HEAD
        if len(l) <= 0: continue
        info = l.split(",")
        if len(info) == 8:
            name, _, _, _, _, y, p, r = info
            is_front = True
        elif len(info) == 9:
            name, is_front, _, _, _, _, y, p, r = info
            is_front = int(is_front) == 1
        else: continue
        if not is_front: continue
        path = os.path.join(data_dir, name)
        if not os.path.exists(path): 
            continue
        data['image'].append(name)

        angles = [float(y), float(p), float(r)]
        data['pose'].append(angles)

    return data

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
    assert view_mode in ["mat", "room", "all"]
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
        p3d = [float(x) for x in [x3d, y3d, z3d]]
        name = name.replace("S", "")
        sess, subj, frame, view = name.replace(".png", "").split("_")
        view = int(view)
        
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
        if type_views == "c" and view_mode == "room": continue
        elif type_views != "c" and view_mode == "mat": continue
        if view_mode == "mat" and not view in [0,1,2,3,6]: continue
        path = os.path.join(data_dir, name)
        angles = [float(y), float(p), float(r)]
        is_corrected = True if int(is_corrected) == 0 else False
        if not os.path.exists(path) or not is_corrected: 
            continue
        
        img = "{}_{}_{}".format(sess, subj, frame)
        if not img in data: 
            data[img] = {"id":count, "name":img}
            count += 1
            
        data[img][view] = {"path":path, "is_front":int(is_front), "dir":angles, "frame":int(frame), "type":type_views,
                           "p3d":p3d}
    #print(data.keys())
    #exit()
=======
        name, _, _, _, _, y, p, r = l.split(",")
        data['image'].append(name)
        angles = [float(y), float(p), float(r)]
        data['pose'].append(angles)
>>>>>>> origin/main
    return data
    
class AFLW2000(Dataset):
    def __init__(self, data_dir, filename_path, transform, img_ext='.jpg', annot_ext='.mat', image_mode='RGB'):
        self.data_dir = data_dir
        self.transform = transform
        self.img_ext = img_ext
        self.annot_ext = annot_ext
        filename_list = get_list_from_filenames(filename_path)

        self.X_train = filename_list
        self.y_train = filename_list
        self.image_mode = image_mode
        self.length = len(filename_list)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.data_dir, self.X_train[index] + self.img_ext))
        img = img.convert(self.image_mode)
        mat_path = os.path.join(self.data_dir, self.y_train[index] + self.annot_ext)

        # Crop the face loosely
        pt2d = utils.get_pt2d_from_mat(mat_path)

        x_min = min(pt2d[0,:])
        y_min = min(pt2d[1,:])
        x_max = max(pt2d[0,:])
        y_max = max(pt2d[1,:])

        k = 0.20
        x_min -= 2 * k * abs(x_max - x_min)
        y_min -= 2 * k * abs(y_max - y_min)
        x_max += 2 * k * abs(x_max - x_min)
        y_max += 0.6 * k * abs(y_max - y_min)
        img = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))

        # We get the pose in radians
        pose = utils.get_ypr_from_mat(mat_path)
        # And convert to degrees.
        pitch = pose[0]# * 180 / np.pi
        yaw = pose[1] #* 180 / np.pi
        roll = pose[2]# * 180 / np.pi
     
        R = utils.get_R(pitch, yaw, roll)

        labels = torch.FloatTensor([yaw, pitch, roll])


        if self.transform is not None:
            img = self.transform(img)

        return img, torch.FloatTensor(R), labels, self.X_train[index]

    def __len__(self):
        # 2,000
        return self.length


class AFLW(Dataset):
    def __init__(self, data_dir, filename_path, transform, img_ext='.jpg', annot_ext='.txt', image_mode='RGB'):
        self.data_dir = data_dir
        self.transform = transform
        self.img_ext = img_ext
        self.annot_ext = annot_ext

        filename_list = get_list_from_filenames(filename_path)

        self.X_train = filename_list
        self.y_train = filename_list
        self.image_mode = image_mode
        self.length = len(filename_list)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.data_dir, self.X_train[index] + self.img_ext))
        img = img.convert(self.image_mode)
        txt_path = os.path.join(self.data_dir, self.y_train[index] + self.annot_ext)

        # We get the pose in radians
        annot = open(txt_path, 'r')
        line = annot.readline().split(' ')
        pose = [float(line[1]), float(line[2]), float(line[3])]
        # And convert to degrees.
        yaw = pose[0] * 180 / np.pi
        pitch = pose[1] * 180 / np.pi
        roll = pose[2] * 180 / np.pi
        # Fix the roll in AFLW
        roll *= -1
        # Bin values
        bins = np.array(range(-99, 102, 3))
        labels = torch.LongTensor(np.digitize([yaw, pitch, roll], bins) - 1)
        cont_labels = torch.FloatTensor([yaw, pitch, roll])

        if self.transform is not None:
            img = self.transform(img)

        return img, labels, cont_labels, self.X_train[index]

    def __len__(self):
        # train: 18,863
        # test: 1,966
        return self.length

class AFW(Dataset):
    def __init__(self, data_dir, filename_path, transform, img_ext='.jpg', annot_ext='.txt', image_mode='RGB'):
        self.data_dir = data_dir
        self.transform = transform
        self.img_ext = img_ext
        self.annot_ext = annot_ext

        filename_list = get_list_from_filenames(filename_path)

        self.X_train = filename_list
        self.y_train = filename_list
        self.image_mode = image_mode
        self.length = len(filename_list)

    def __getitem__(self, index):
        txt_path = os.path.join(self.data_dir, self.y_train[index] + self.annot_ext)
        img_name = self.X_train[index].split('_')[0]

        img = Image.open(os.path.join(self.data_dir, img_name + self.img_ext))
        img = img.convert(self.image_mode)
        txt_path = os.path.join(self.data_dir, self.y_train[index] + self.annot_ext)

        # We get the pose in degrees
        annot = open(txt_path, 'r')
        line = annot.readline().split(' ')
        yaw, pitch, roll = [float(line[1]), float(line[2]), float(line[3])]

        # Crop the face loosely
        k = 0.32
        x1 = float(line[4])
        y1 = float(line[5])
        x2 = float(line[6])
        y2 = float(line[7])
        x1 -= 0.8 * k * abs(x2 - x1)
        y1 -= 2 * k * abs(y2 - y1)
        x2 += 0.8 * k * abs(x2 - x1)
        y2 += 1 * k * abs(y2 - y1)

        img = img.crop((int(x1), int(y1), int(x2), int(y2)))

        # Bin values
        bins = np.array(range(-99, 102, 3))
        labels = torch.LongTensor(np.digitize([yaw, pitch, roll], bins) - 1)
        cont_labels = torch.FloatTensor([yaw, pitch, roll])

        if self.transform is not None:
            img = self.transform(img)

        return img, labels, cont_labels, self.X_train[index]

    def __len__(self):
        # Around 200
        return self.length

class BIWI(Dataset):
    def __init__(self, data_dir, filename_path, transform, image_mode='RGB', train_mode=True):
        self.data_dir = data_dir
        self.transform = transform

        d = np.load(filename_path)

        x_data = d['image']
        y_data = d['pose']
        self.X_train = x_data
        self.y_train = y_data
        self.image_mode = image_mode
        self.train_mode = train_mode
        self.length = len(x_data)

    def __getitem__(self, index):
        img = Image.fromarray(np.uint8(self.X_train[index]))
        img = img.convert(self.image_mode)

        roll = self.y_train[index][2]/180*np.pi
        yaw = self.y_train[index][0]/180*np.pi
        pitch = self.y_train[index][1]/180*np.pi
        cont_labels = torch.FloatTensor([yaw, pitch, roll])

        if self.train_mode:
            # Flip?
            rnd = np.random.random_sample()
            if rnd < 0.5:
                yaw = -yaw
                roll = -roll
                img = img.transpose(Image.FLIP_LEFT_RIGHT)

            # Blur?
            rnd = np.random.random_sample()
            if rnd < 0.05:
                img = img.filter(ImageFilter.BLUR)

        R = utils.get_R(pitch, yaw, roll)

        labels = torch.FloatTensor([yaw, pitch, roll])

        if self.transform is not None:
            img = self.transform(img)


        # Get target tensors
        cont_labels = torch.FloatTensor([yaw, pitch, roll])
        return img, torch.FloatTensor(R), cont_labels, self.X_train[index]

    def __len__(self):
        # 15,667
        return self.length

class Pose_300W_LP(Dataset):
    # Head pose from 300W-LP dataset
    def __init__(self, data_dir, filename_path, transform, img_ext='.jpg', annot_ext='.mat', image_mode='RGB'):
        self.data_dir = data_dir
        self.transform = transform
        self.img_ext = img_ext
        self.annot_ext = annot_ext
        filename_list = get_list_from_filenames(filename_path)

        self.X_train = filename_list
        self.y_train = filename_list
        self.image_mode = image_mode
        self.length = len(filename_list)

    def __getitem__(self, index):
        img = Image.open(os.path.join(
            self.data_dir, self.X_train[index] + self.img_ext))
        img = img.convert(self.image_mode)
        mat_path = os.path.join(
            self.data_dir, self.y_train[index] + self.annot_ext)

        # Crop the face loosely
        pt2d = utils.get_pt2d_from_mat(mat_path)
        x_min = min(pt2d[0, :])
        y_min = min(pt2d[1, :])
        x_max = max(pt2d[0, :])
        y_max = max(pt2d[1, :])

        # k = 0.2 to 0.40
        k = np.random.random_sample() * 0.2 + 0.2
        x_min -= 0.6 * k * abs(x_max - x_min)
        y_min -= 2 * k * abs(y_max - y_min)
        x_max += 0.6 * k * abs(x_max - x_min)
        y_max += 0.6 * k * abs(y_max - y_min)
        img = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))

        # We get the pose in radians
        pose = utils.get_ypr_from_mat(mat_path)
        # And convert to degrees.
        pitch = pose[0] # * 180 / np.pi
        yaw = pose[1] #* 180 / np.pi
        roll = pose[2] # * 180 / np.pi

        # Gray images

        # Flip?
        rnd = np.random.random_sample()
        if rnd < 0.5:
            yaw = -yaw
            roll = -roll
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        # Blur?
        rnd = np.random.random_sample()
        if rnd < 0.05:
            img = img.filter(ImageFilter.BLUR)

        # Add gaussian noise to label
        #mu, sigma = 0, 0.01 
        #noise = np.random.normal(mu, sigma, [3,3])
        #print(noise) 

        # Get target tensors
        R = utils.get_R(pitch, yaw, roll)#+ noise
        
        #labels = torch.FloatTensor([temp_l_vec, temp_b_vec, temp_f_vec])

        if self.transform is not None:
            img = self.transform(img)

        return img,  torch.FloatTensor(R),[], self.X_train[index]

    def __len__(self):
        # 122,450
        return self.length

class DevEv(Dataset):
    def __init__(self, data_dir, filename_path, transform, image_mode='RGB', train_mode=True):
        self.data_dir = data_dir
        self.transform = transform

<<<<<<< HEAD
        d = read_gt(filename_path, data_dir, train_mode)

        x_data = d['image']
        y_data = d['pose']
        self.X_train = x_data
        self.y_train = y_data
        self.image_mode = image_mode
        self.train_mode = train_mode
        self.length = len(x_data)

    def __getitem__(self, index):
        name = os.path.join(self.data_dir, self.X_train[index])
        img = Image.open(name)
        img = img.convert(self.image_mode)
        
        if "set_" not in name:
            width, height = img.size

            # Define the percentages for cropping
            xmin = 0.25 * width
            ymin = 0.25 * height
            xmax = 0.75 * width
            ymax = 0.75 * height

            # Cropped image using percentages
            img = img.crop((xmin, ymin, xmax, ymax))

        roll = self.y_train[index][2]/180*np.pi
        yaw = self.y_train[index][0]/180*np.pi
        pitch = self.y_train[index][1]/180*np.pi
        cont_labels = torch.FloatTensor([yaw, pitch, roll])

        if self.train_mode:
            # Flip?
            rnd = np.random.random_sample()
            #if rnd < 0.5:
            #    yaw = -yaw
            #    roll = -roll
            #    img = img.transpose(Image.FLIP_LEFT_RIGHT)

            # Blur?
            rnd = np.random.random_sample()
            if rnd < 0.05:
                img = img.filter(ImageFilter.BLUR)

        R = utils.get_R(pitch, yaw, roll)

        labels = torch.FloatTensor([yaw, pitch, roll])

        if self.transform is not None:
            img = self.transform(img)


        # Get target tensors
        cont_labels = torch.FloatTensor([yaw, pitch, roll])
        return img, torch.FloatTensor(R), cont_labels, self.X_train[index]

    def __len__(self):
        # 15,667
        return self.length

class DevEv3D(Dataset):
    def __init__(self, data_dir, filename_path, transform, image_mode='RGB', train_mode=True):
        self.data_dir = data_dir
        self.transform = transform

=======
>>>>>>> origin/main
        d = read_gt(filename_path)

        x_data = d['image']
        y_data = d['pose']
        self.X_train = x_data
        self.y_train = y_data
        self.image_mode = image_mode
        self.train_mode = train_mode
        self.length = len(x_data)

    def __getitem__(self, index):
        img = Image.open(os.path.join(
            self.data_dir, self.X_train[index]))
        img = img.convert(self.image_mode)

        roll = self.y_train[index][2]/180*np.pi
        yaw = self.y_train[index][0]/180*np.pi
        pitch = self.y_train[index][1]/180*np.pi
<<<<<<< HEAD
        cont_labels = torch.FloatTensor([yaw, pitch, roll])
=======
>>>>>>> origin/main

        if self.train_mode:
            # Flip?
            rnd = np.random.random_sample()
            if rnd < 0.5:
                yaw = -yaw
                roll = -roll
                img = img.transpose(Image.FLIP_LEFT_RIGHT)

            # Blur?
            rnd = np.random.random_sample()
            if rnd < 0.05:
                img = img.filter(ImageFilter.BLUR)
<<<<<<< HEAD

        R = utils.get_R(pitch, yaw, roll)

        labels = torch.FloatTensor([yaw, pitch, roll])

=======
                
        roll = roll *0.0
        
        R = utils.get_R(pitch, yaw, roll)
>>>>>>> origin/main
        if self.transform is not None:
            img = self.transform(img)


        # Get target tensors
        cont_labels = torch.FloatTensor([yaw, pitch, roll])
        return img, torch.FloatTensor(R), cont_labels, self.X_train[index]

    def __len__(self):
        # 15,667
        return self.length

<<<<<<< HEAD
def get_quadrant(gt, num_quadrants):
    # Assuming gt is a 3D normalized vector
    gt = gt/np.linalg.norm(gt)
    x, y, z = gt
    if np.sqrt(x**2+y**2) == 0: theta = 0.0
    else: theta = np.arccos(x/np.sqrt(x**2+y**2)) + np.pi*int(y<0)
    if np.sqrt(y**2+z**2) == 0: phi = 0.0
    else: phi = np.arccos(z)

    # Map the angle to the corresponding quadrant
    theta_quadrant = int(np.floor(num_quadrants * theta / (2 * np.pi) )) #% num_quadrants
    phi_quadrant = int(np.floor(num_quadrants * phi /np.pi)) #% num_quadrants

    # Combine the two angles to determine the final quadrant
    return theta_quadrant * num_quadrants + phi_quadrant

class DevEvMV(Dataset):
    def __init__(self, data_dir, filename_path, transform, image_mode='RGB', train_mode=True):
        self.data_dir = data_dir
        self.transform = transform

        mode = "room"
        mode = "mat"
        d = read_gt_mv(filename_path, data_dir, view_mode = mode)
        self.X_train = {}
        count, count_test = 0, 0

        with open("test_data.txt", "r") as f:
            test_data = [x.replace("\n","") for x in f.readlines()]
        if mode == "mat": test_data = []
        for i, (n, info) in enumerate(d.items()):
            if mode == "mat":
                if i < 0.2*len(d) and not train_mode: 
                    self.X_train[count_test] = info
                    count_test += 1
                elif train_mode: 
                    self.X_train[count] = info
                    count += 1
                continue
            else:
                if n in test_data and not train_mode: 
                    self.X_train[count_test] = info
                    count_test += 1
                elif train_mode and n not in test_data: 
                    self.X_train[count] = info
                    count += 1

        if train_mode:
            d = read_gt_mv(filename_path.replace("_new","_quad"), data_dir.replace("_new","_quad"), view_mode = mode)
            for i, (n, info) in enumerate(d.items()):
                self.X_train[count] = info
                count += 1

 
        num_quad = 4
        quad_hist = {i:[] for i in range(num_quad*num_quad)}
        for index, info in self.X_train.items():
            views = [x for x in list(info.keys()) if type(x)==int]
            gt = info[views[0]]["dir"]
            quad = get_quadrant(gt, num_quad)
            #print(gt, quad)
            quad_hist[quad].append(index)

        
        """
        with open("data_imbalance.txt", "w") as f: f.write("")
        for i, d in quad_hist.items():
            print(i, len(d))
            #if len(d) >= 20: continue
            d_selected = d[0:2]
            for index in d_selected:
                with open("data_imbalance.txt", "a") as f: 
                    f.write("Quadrant_{},{},{}\n".format(i, len(d), self.X_train[index]['name']))
        exit()"""
  
        """new_data, count, N = {}, 0, 300
        N_test = 50
        blacklist = [0,1,4,5,8,9,12,13]
        for i, d in quad_hist.items():
            if len(d) == 0: continue
            if i in blacklist: continue
            selection = d
            
            #if train_mode:
            #    if len(d) >= N:
            #        selection = [d[n] for n in np.linspace(0, len(d)-1, N, dtype=int)]
            #else:
            #    if len(d) >= N_test:
            #        selection = [d[n] for n in np.linspace(0, len(d)-1, N_test, dtype=int)]
            #print(i, len(d), len(selection))
            for s in selection: 
                new_data[count] = self.X_train[s]
                new_data[count]["quad"] = i
                count += 1
                
        self.X_train = new_data """
   
            
        self.image_mode = image_mode
        self.train_mode = train_mode
        self.length = len(self.X_train.keys())
        print(self.length, "Train:", train_mode)
     
        
    def __getitem__(self, index):
        info = self.X_train[index]
        img_list, views = torch.zeros((8,3,224,224)).float(), torch.zeros((8)).long()
        #img_list, views = torch.zeros((8,3,256,256)).float(), torch.zeros((8)).long()
        for i in range(8):
            if i not in info: continue
            img = Image.open(os.path.join(info[i]["path"]))
            
            img = img.convert(self.image_mode)


            if self.train_mode:
                # Flip?
                #rnd = np.random.random_sample()
                #if rnd < 0.5:
                #    yaw = -yaw
                #    roll = -roll
                #    img = img.transpose(Image.FLIP_LEFT_RIGHT)

                # Blur?
                rnd = np.random.random_sample()
                if rnd < 0.05:
                    img = img.filter(ImageFilter.BLUR)
            if self.transform is not None:
                img = self.transform(img)
            img_list[i] = img
            views[i] = 1
            att_dir = info[i]["dir"]

        R = utils.rotation_matrix_from_vectors(np.array([0,0,1.0]), att_dir)

        # Get target tensors
        cont_labels = torch.FloatTensor([att_dir[0], att_dir[1], att_dir[2]])
        return img_list, torch.FloatTensor(R), cont_labels, views

    def __len__(self):
        # 15,667
        return self.length

   
=======
>>>>>>> origin/main
def getDataset(dataset, data_dir, filename_list, transformations, train_mode = True):
    if dataset == 'Pose_300W_LP':
            pose_dataset = Pose_300W_LP(
                data_dir, filename_list, transformations)
    elif dataset == 'AFLW2000':
        pose_dataset = AFLW2000(
            data_dir, filename_list, transformations)
    elif dataset == 'BIWI':
        pose_dataset = BIWI(
            data_dir, filename_list, transformations, train_mode= train_mode)
    elif dataset == 'AFLW':
        pose_dataset = AFLW(
            data_dir, filename_list, transformations)
    elif dataset == 'AFW':
        pose_dataset = AFW(
            data_dir, filename_list, transformations)
    elif dataset == 'DevEv':
        pose_dataset = DevEv(
<<<<<<< HEAD
            data_dir, filename_list, transformations, train_mode = train_mode)
    elif dataset == 'DevEvMV':
        pose_dataset = DevEvMV(
            data_dir, filename_list, transformations, train_mode = train_mode)
    elif dataset == 'DevEv3D':
        pose_dataset = DevEv(
            data_dir, filename_list, transformations, train_mode = train_mode)
=======
            data_dir, filename_list, transformations)
>>>>>>> origin/main
    else:
        raise NameError('Error: not a valid dataset name')

    return pose_dataset
