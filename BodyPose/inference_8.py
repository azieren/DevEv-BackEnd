from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision
import cv2
import numpy as np
import time

from .model import get_pose_net
from .config import _C as cfg
from .config import update_config
from .utils import get_affine_transform, get_final_preds, place_camera

from .metadata import coco_metadata, COCO_KEYPOINT_INDEXES, COCO_INSTANCE_CATEGORY_NAMES, SKELETON, CocoColors

NUM_KPTS = 17

PARENT_BOX_B_1 = [685, 418, 960, 540]
PARENT_BOX_B_2 = [314, 0, 500, 127]

PARENT_BOX_M_1 = [175, 49, 251, 127]
PARENT_BOX_M_2 = [619, 115, 683, 184]

CURR_FOLDER = os.path.dirname(os.path.abspath(__file__))

CTX = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def draw_pose(keypoints,img):
    """draw the keypoints and the skeletons.
    :params keypoints: the shape should be equal to [17,2]
    :params img:
    """
    assert keypoints.shape == (NUM_KPTS,2)
    for i in range(len(SKELETON)):
        #if i == 14 : 
        kpt_a, kpt_b = SKELETON[i][0], SKELETON[i][1]
        x_a, y_a = keypoints[kpt_a][0],keypoints[kpt_a][1]
        x_b, y_b = keypoints[kpt_b][0],keypoints[kpt_b][1] 
        cv2.circle(img, (int(x_a), int(y_a)), 6, CocoColors[i], -1)
        cv2.circle(img, (int(x_b), int(y_b)), 6, CocoColors[i], -1)
        cv2.line(img, (int(x_a), int(y_a)), (int(x_b), int(y_b)), CocoColors[i], 2)

def draw_bbox(box,img):
    """draw the detected bounding box on the image.
    :param img:
    """
    cv2.rectangle(img, box[0], box[1], color=(0, 255, 0),thickness=3)


def get_person_detection_boxes(model, img, threshold=0.5):
    pred = model(img)
    all_person_boxes = []
    for p in range(len(pred)):
        pred_classes = [COCO_INSTANCE_CATEGORY_NAMES[i]
                        for i in list(pred[p]['labels'].cpu().numpy())]  # Get the Prediction Score
        pred_boxes = [[(i[0], i[1]), (i[2], i[3])]
                    for i in list(pred[p]['boxes'].detach().cpu().numpy())]  # Bounding boxes
        pred_score = list(pred[p]['scores'].detach().cpu().numpy())
        if not pred_score or max(pred_score)<threshold:
            all_person_boxes.append([])
            continue
        # Get list of index with score greater than threshold
        pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1]
        pred_boxes = pred_boxes[:pred_t+1]
        pred_classes = pred_classes[:pred_t+1]

        person_boxes = []
        for idx, box in enumerate(pred_boxes):
            if pred_classes[idx] == 'person':
                person_boxes.append(box)
        all_person_boxes.append(person_boxes)
    return all_person_boxes

def process_bbox(image, center, scale):
    rotation = 0
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # pose estimation transformation
    trans = get_affine_transform(center, scale, rotation, cfg.MODEL.IMAGE_SIZE)
    bbox = cv2.warpAffine(
        image,
        trans,
        (int(cfg.MODEL.IMAGE_SIZE[0]), int(cfg.MODEL.IMAGE_SIZE[1])),
        flags=cv2.INTER_LINEAR)

    return transform(bbox)

def get_pose_estimation_prediction(pose_model, input_model, center, scale):

    # switch to evaluate mode
    input_model = torch.stack(input_model, 0)
    pose_model.eval()
    with torch.no_grad():
        # compute output heatmap
        output = pose_model(input_model)
        preds, _ = get_final_preds(
            cfg,
            output.clone().cpu().numpy(),
            np.asarray(center),
            np.asarray(scale))

        return preds

def box_to_center_scale(box, model_image_width, model_image_height):
    """convert a box to center,scale information required for pose transformation
    Parameters
    ----------
    box : list of tuple
        list of length 2 with two tuples of floats representing
        bottom left and top right corner of a box
    model_image_width : int
    model_image_height : int

    Returns
    -------
    (numpy array, numpy array)
        Two numpy arrays, coordinates for the center of the box and the scale of the box
    """
    center = np.zeros((2), dtype=np.float32)

    bottom_left_corner = box[0]
    top_right_corner = box[1]
    box_width = top_right_corner[0]-bottom_left_corner[0]
    box_height = top_right_corner[1]-bottom_left_corner[1]
    bottom_left_x = bottom_left_corner[0]
    bottom_left_y = bottom_left_corner[1]
    center[0] = bottom_left_x + box_width * 0.5
    center[1] = bottom_left_y + box_height * 0.5

    aspect_ratio = model_image_width * 1.0 / model_image_height
    pixel_std = 200

    if box_width > aspect_ratio * box_height:
        box_height = box_width * 1.0 / aspect_ratio
    elif box_width < aspect_ratio * box_height:
        box_width = box_height * aspect_ratio
    scale = np.array(
        [box_width * 1.0 / pixel_std, box_height * 1.0 / pixel_std],
        dtype=np.float32)
    if center[0] != -1:
        scale = scale * 1.25

    return center, scale


def main_bodypose(videofile, timestamps, write=True, output_dir = None):
    if not os.path.exists(videofile):
        raise Exception("Video file {} not found".format(videofile))

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    update_config(cfg, os.path.join(CURR_FOLDER, "config.yaml"))

    box_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    box_model.to(CTX)
    box_model.eval()

    pose_model = get_pose_net(
        cfg, is_train=False
    )

    if cfg.TEST.MODEL_FILE:
        print('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
        pose_model.load_state_dict(torch.load(os.path.join(CURR_FOLDER, cfg.TEST.MODEL_FILE)), strict=False)
    else:
        print('expected model defined in config at TEST.MODEL_FILE')

    pose_model = torch.nn.DataParallel(pose_model, device_ids=cfg.GPUS)
    pose_model.to(CTX)
    pose_model.eval()

    # Loading an video or an image or webcam 
    if videofile:
        vidcap = cv2.VideoCapture(videofile)
    else:
        print('please use --video or --webcam or --image to define the input.')
        return 

    name = videofile.split("/")[-1]
    base_name = name.split(".")[0]
    
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    h, w = int(vidcap.get(4)), int(vidcap.get(3))

    if output_dir is None:
        output_dir = os.path.join(CURR_FOLDER, "output")
    if write:
        save_path = os.path.join(output_dir, 'output_' + name)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        #out = cv2.VideoWriter(save_path,fourcc, 12.0, (int(vidcap.get(3)//2),int(vidcap.get(4)//2)))
        out = cv2.VideoWriter(save_path,fourcc, fps, (w, h))
    with open(os.path.join(output_dir,'output_' + base_name + ".txt"), "w") as f: 
        f.write("")

    starts, ends = [], []
    for n, info in timestamps.items():
        for (s,e) in info:
            starts.append(int(fps*s/1000))
            ends.append(int(fps*e/1000))

    if len(timestamps) == 0:
        starts, ends = [0], [np.inf]
    else:
        index = np.argsort(starts)
        starts, ends = np.array(starts), np.array(ends)
        starts = starts[index]
        ends = ends[index]
    
    current_segment = 0
    count = starts[0]
    vidcap.set(1, count)
    dataset_kpt = {}
    while True:
        
        if count > ends[current_segment]:
            current_segment += 1
            if current_segment < len(starts):
                count = starts[current_segment]
                vidcap.set(1, count)
            else:
                break


        ret, image_bgr = vidcap.read()
        
        
        if ret:
            #image_bgr = image_bgr[:h//2, w//2:]
            last_time = time.time()
            image = image_bgr[:, :, [2, 1, 0]]
            
            img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            img_tensor = torch.from_numpy(img/255.).permute(2,0,1).float().to(CTX)
            img_tensor1, img_tensor2 = img_tensor[:, :h//4, :w//2], img_tensor[:, :h//4, w//2:]
            img_tensor3, img_tensor4 = img_tensor[:,h//4:h//2,:w//2], img_tensor[:,h//4:h//2 ,w//2:]
            img_tensor5, img_tensor6 = img_tensor[:,h//2:3*h//4,:w//2], img_tensor[:,h//2:3*h//4,w//2:]
            img_tensor7, img_tensor8 = img_tensor[:,3*h//4:,:w//2], img_tensor[:,3*h//4:,w//2:]
            #print(img_tensor.size(), img_tensor1.size())
            #exit()
            input = [img_tensor1, img_tensor2, img_tensor3, img_tensor4, img_tensor5, img_tensor6, img_tensor7, img_tensor8]


            # object detection box
            with torch.no_grad():
                pred_boxes = get_person_detection_boxes(box_model, input, threshold=0.80)

            # pose estimation
            #dataset_kpt[count] = {"views":[], "bbox":[], "kpt":[]}
            if len(pred_boxes) >= 1:
                bbox_tensor = []
                kps = []
                views = []
                
                image_pose = image.copy() if cfg.DATASET.COLOR_RGB else image_bgr.copy()
                center_list, scale_list, box_list, box_viz = [], [], [], [] 
                view_id_list = []
                tensor_list = []
                for im_id in range(len(pred_boxes)):
                    left, right = 0, 0
                    if im_id == 0: img = image_pose[:h//4, :w//2] 
                    elif im_id == 1: 
                        left = w//2
                        img = image_pose[:h//4, w//2:] 
                    elif im_id == 2: 
                        right = h//4
                        img = image_pose[h//4:h//2, :w//2]
                    elif im_id == 3: 
                        left, right = w//2, h//4
                        img = image_pose[h//4:h//2, w//2:] 
                    elif im_id == 4: 
                        right = h//2
                        img = image_pose[h//2:3*h//4, :w//2]
                    elif im_id == 5: 
                        left, right = w//2, h//2
                        img = image_pose[h//2:3*h//4, w//2:] 
                    elif im_id == 6: 
                        right = 3*h//4
                        img = image_pose[3*h//4:, :w//2] 
                    elif im_id == 7: 
                        left, right = w//2, 3*h//4
                        img = image_pose[3*h//4:, w//2:] 

                    for b_id, box in enumerate(pred_boxes[im_id]):
                        center, scale = box_to_center_scale(box, cfg.MODEL.IMAGE_SIZE[0], cfg.MODEL.IMAGE_SIZE[1])
                        center_list.append(center) 
                        scale_list.append(scale) 
                        box_list.append(np.copy([box[0][0] + left, box[0][1]+ right, box[1][0]+ left, box[1][1] + right, 1.0]))
                        box_viz.append(np.copy([box[0][0], box[0][1], box[1][0], box[1][1], 1.0]))
                        tensor_list.append(process_bbox(img, center, scale) )
                        view_id_list.append(im_id)

                box_list = np.array(box_list).astype("int")
                box_viz = np.copy(box_list).astype("int")
                
                if len(tensor_list) > 0: 
                    pose_preds = get_pose_estimation_prediction(pose_model, tensor_list, center_list, scale_list)
                    pose_viz = place_camera(view_id_list, np.copy(pose_preds), h, w)

                    for i in range(len(box_list)):
                        im_id = view_id_list[i]
                        x, y = pose_preds[i,0,0], pose_preds[i,0,1]
                        #if im_id in [1] and (PARENT_BOX_M_1[0]  <= x <= PARENT_BOX_M_1[2] and PARENT_BOX_M_1[1] <= y <= PARENT_BOX_M_1[3]) : continue
                        #if im_id in [2] and (PARENT_BOX_M_2[0]  <= x <= PARENT_BOX_M_2[2] and PARENT_BOX_M_2[1] <= y <= PARENT_BOX_M_2[3]) : continue

                        #if im_id in [5] and (PARENT_BOX_B_1[0]  <= x <= PARENT_BOX_B_1[2] and PARENT_BOX_B_1[1] <= y <= PARENT_BOX_B_1[3]) : continue
                        #if im_id in [6] and (PARENT_BOX_B_2[0]  <= x <= PARENT_BOX_B_2[2] and PARENT_BOX_B_2[1] <= y <= PARENT_BOX_B_2[3]) : continue
                        #if im_id in [5] and (y <= 90) : continue

                        kps_logit = np.zeros((pose_preds[i].shape[0], 2))
                        kps_logit[:,1] = 1.0
                        kps_all = np.concatenate((pose_preds[i], kps_logit), axis=1)

                        bbox_tensor.append(box_list[i])
                        views.append(im_id)
                        kps.append(np.copy(kps_all))

                        if write:
                            draw_bbox([(box_viz[i,0], box_viz[i,1]), (box_viz[i,2], box_viz[i,3])], image_bgr)
                            draw_pose(pose_viz[i], image_bgr) # draw the poses
                        with open(os.path.join(output_dir,'output_' + base_name + ".txt"), "a") as f: 
                            f.write("{},{},{},{},{},{},{},{},{},{},{}\n".format(count,0, b_id, 
                                box_list[i,0], box_list[i,1], box_list[i,2], box_list[i,3], pose_preds[i,15,0], pose_preds[i,15,1], pose_preds[i,0,0], pose_viz[i,0,1]))
                    
                dataset_kpt[count] = {"views":views, "bbox":bbox_tensor, "kpt":kps}

            if write:
                out.write(image_bgr)
            count += 1

            if count % 50 == 0:
                print("Segment {}/{} - frame {} - Start frame {} - End frame {}".format(current_segment+1, len(starts), 
                    count, starts[current_segment], ends[current_segment]))

        else:
            print('cannot load the video.')
            break

    #cv2.destroyAllWindows()
    vidcap.release()
    if write:
        print('video has been saved as {}'.format(save_path))
        out.release()
    # Video resolution
    metadata_ = {
        'w': w,
        'h': h,
    }
    #np.savez_compressed('npz_' + base_name , boxes=boxes, segments=segments, keypoints=keypoints, metadata=metadata)
    #data, video_metadata = decode(data)

    metadata = coco_metadata
    metadata['video_metadata'] = metadata_
    metadata["keypoints_name"] = COCO_KEYPOINT_INDEXES
    metadata["video"] = base_name
    np.savez_compressed(os.path.join(output_dir,'data_2d_' + base_name), data=dataset_kpt, metadata=metadata)



if __name__ == '__main__':
    video = "../DevEv_31_03 (2).mp4"
    timestamps = {"c":(0, 1100), "p":(1200, 3000), "d":(3100, 5000)}
    main_bodypose(video, timestamps, write = True)
